"""lmms-eval model: LLaVA-1.5 with QViD pruning.

Registered as `--model qvid_llava` through the `lmms_eval.models` entry point
(see pyproject.toml), e.g.

    python -m lmms_eval --model qvid_llava \
        --model_args pretrained=liuhaotian/llava-v1.5-7b,num_tokens=64 \
        --tasks pope,gqa --batch_size 1

num_tokens=576 disables pruning (vanilla LLaVA-1.5).
"""

from __future__ import annotations

from typing import List

import PIL
import torch
from loguru import logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

try:
    from lmms_eval.models.registry_v2 import ModelManifest

    # Picked up by lmms-eval through the `lmms_eval.models` entry point in pyproject.toml.
    MANIFEST = ModelManifest(model_id="qvid_llava",
                             simple_class_path="llava.eval.lmms_eval_qvid.QViDLlava")
except ImportError:  # lmms-eval without the v2 registry
    MANIFEST = None


class QViDLlava(lmms):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        num_tokens: int = 64,
        scoring_layer: int = 2,
        num_components: int = 16,
        conv_template: str = "vicuna_v1",
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"Ignoring unknown model_args: {sorted(kwargs)}")

        self._device = device
        self.num_tokens = int(num_tokens)
        self.conv_template = conv_template

        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=pretrained,
            model_base=None,
            model_name=get_model_name_from_path(pretrained),
            device=device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.get_model().set_qvid(num_tokens=self.num_tokens, layer=int(scoring_layer),
                                        num_components=int(num_components))

        self._rank = 0
        self._world_size = 1

    # -- lmms-eval interface ---------------------------------------------

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until_multi_round(self, requests):
        raise NotImplementedError

    # -- generation ---------------------------------------------------------

    def _generate(self, image, question: str, max_new_tokens: int, until: list) -> str:
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question if image is None else DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if image is not None:
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self._device)
            images = process_images(
                [image.convert("RGB")], self.image_processor, self.model.config
            ).to(dtype=torch.float16, device=self._device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
            images = None

        stopping = [KeywordsStoppingCriteria(until, self.tokenizer, input_ids)] if until else None
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping,
            )

        # LLaVA's generate() runs on inputs_embeds, so only new tokens are returned.
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        for u in until or []:
            if u in output:
                output = output[:output.index(u)]
        return output.strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(self.tok_encode(x[0])), x[0]

        re_ords = utils.Collator([req.args for req in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=1, batch_fn=None)
        pbar = tqdm(total=len(requests), desc=f"QViD (num_tokens={self.num_tokens})")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            visuals = doc_to_visual[0](self.task_dict[task][split][doc_id[0]]) or []
            if not isinstance(visuals, list):
                visuals = [visuals]

            context = contexts[0]
            gen_kwargs = all_gen_kwargs[0]
            max_new = gen_kwargs.get("max_new_tokens", 128)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]

            # "ASSISTANT:" belongs to the vicuna_v1 template and never appears in
            # generated text. When it is the only stop string on a short-answer
            # task, stop at the first newline instead so heavily pruned models
            # cannot ramble. Tasks with no stop string (e.g. MME) or long answers
            # (max_new_tokens > 128) are left unchanged.
            had_assistant = "ASSISTANT:" in until
            until = [u for u in until if u != "ASSISTANT:"]
            if had_assistant and not until and max_new <= 128:
                until = ["\n"]
                max_new = min(max_new, 32)

            image = visuals[0] if visuals and isinstance(visuals[0], PIL.Image.Image) else None
            if image is not None:
                question = context.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            else:  # text-only sample: no stop strings
                question, until = context, []
            try:
                output = self._generate(image, question, max_new, until)
            except Exception as e:
                logger.error(f"QViD generation failed on {task}/{doc_id[0]}: {e}")
                output = ""

            res.append(output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output)
            pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
