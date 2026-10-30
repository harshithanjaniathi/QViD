"""Answer one question about one image with QViD-pruned LLaVA-1.5.

    python scripts/demo.py --image path/to/image.jpg \
        --question "What is the man holding?" --num-tokens 64
"""

import argparse

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--num-tokens", type=int, default=64,
                        help="visual tokens kept from the scoring layer on (576 = no pruning)")
    parser.add_argument("--scoring-layer", type=int, default=2,
                        help="0-based decoder layer index (2 = the paper's layer 3)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path),
        device=args.device, attn_implementation="sdpa",
    )
    model.get_model().set_qvid(num_tokens=args.num_tokens, layer=args.scoring_layer)

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)
    image = Image.open(args.image).convert("RGB")
    images = process_images([image], image_processor, model.config).to(
        dtype=torch.float16, device=args.device)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=images, do_sample=False,
                                    max_new_tokens=128, use_cache=True)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
