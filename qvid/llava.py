"""QViD for LLaVA-1.5 (haotian-liu/LLaVA, transformers 4.37).

Visual tokens are pruned inside the LLM, at the input of `scoring_layer`:

    layers 0 .. L-1   see all visual tokens (576 for LLaVA-1.5)
    layer  L          pre-hook scores visual tokens with layer L's Q/K
                      projections and keeps the top `num_tokens`
    layers L .. end   see only the kept tokens

Kept tokens retain their original position ids, so RoPE is unchanged for
them and for all subsequent decoding steps.

Usage:
    tokenizer, model, image_processor, _ = load_pretrained_model(
        path, None, name, attn_implementation="sdpa")
    apply_qvid(model, num_tokens=64)
    model.generate(input_ids, images=..., ...)   # pruning happens inside
"""

from __future__ import annotations

import types

import torch
import transformers
from llava.constants import IMAGE_TOKEN_INDEX

from qvid.scoring import qvid_scores, select_topk


class _QViDState:
    """Hooks and RoPE patches that live for a single generate() call."""

    def __init__(self):
        self.handles = []
        self.patched_rotary = []

    def reset(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        for rotary in self.patched_rotary:
            del rotary.forward  # drop the instance override, back to the class method
        self.patched_rotary = []


def apply_qvid(model, num_tokens: int = 64, scoring_layer: int = 3, num_components: int = 16):
    """Enable QViD pruning on a loaded LlavaLlamaForCausalLM (in place).

    Args:
        model: LLaVA-1.5 model loaded with attn_implementation="sdpa".
        num_tokens: visual tokens kept from `scoring_layer` onward.
        scoring_layer: LLM layer at whose input tokens are scored and pruned.
        num_components: rank r of the singular subspace used for scoring.
    """
    if not transformers.__version__.startswith("4.37."):
        raise RuntimeError(
            f"QViD for LLaVA-1.5 requires transformers 4.37.x, found {transformers.__version__}. "
            "Run: pip install transformers==4.37.2 tokenizers==0.15.2"
        )
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise ValueError(
            "QViD requires attn_implementation='sdpa' (pruned layers rely on "
            "SDPA's implicit causal mask). Pass it to load_pretrained_model()."
        )
    if hasattr(model, "_qvid_state"):
        raise RuntimeError("apply_qvid() was already called on this model.")

    llm = model.get_model()
    layers = llm.layers
    num_heads = model.config.num_attention_heads
    num_vis = model.get_vision_tower().num_patches
    if not 0 <= scoring_layer < len(layers):
        raise ValueError(f"scoring_layer must be in [0, {len(layers) - 1}]")

    state = _QViDState()
    model._qvid_state = state
    original_prepare = model.prepare_inputs_labels_for_multimodal
    original_generate = model.generate

    def _patch_rotary():
        # Kept tokens keep their original position ids (up to the full
        # sequence length), but after pruning the per-layer kv length is
        # shorter, and 4.37's rotary embedding slices its cos/sin tables to
        # that length. Return the full tables so original positions index
        # correctly in the pruned prefill and in every decoding step.
        for layer in layers[scoring_layer:]:
            rotary = layer.self_attn.rotary_emb

            def _full_table(x, seq_len=None, _rotary=rotary):
                return (_rotary.cos_cached.to(dtype=x.dtype),
                        _rotary.sin_cached.to(dtype=x.dtype))

            rotary.forward = _full_table
            state.patched_rotary.append(rotary)

    def _pruned_prepare(self, input_ids, position_ids, attention_mask,
                        past_key_values, labels, images, image_sizes=None):
        if input_ids.shape[1] > 1:  # new prefill: drop hooks left from a previous forward
            state.reset()

        result = original_prepare(input_ids, position_ids, attention_mask,
                                  past_key_values, labels, images, image_sizes)
        inputs_embeds = result[4]
        if inputs_embeds is None or num_tokens >= num_vis:
            return result
        if input_ids.shape[0] != 1:
            raise NotImplementedError("QViD supports batch size 1 only.")

        img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if img_pos.numel() != 1:  # no image, or multiple images
            return result
        vis_start = int(img_pos[0])
        vis_end = vis_start + num_vis
        if inputs_embeds.shape[1] < vis_end:
            return result

        fired = [False]
        pruned_pos = [None]

        def _score_and_prune(module, args, kwargs):
            hs = args[0]
            if fired[0] or hs.shape[1] <= 1:  # already pruned, or a decoding step
                return None
            fired[0] = True

            # Text = everything after the image (question, instruction, "ASSISTANT:").
            vis_hs = hs[0, vis_start:vis_end]
            txt_hs = hs[0, vis_end:]
            if txt_hs.shape[0] == 0:
                txt_hs = hs[0, :vis_start]

            ln, attn = module.input_layernorm, module.self_attn
            with torch.no_grad():
                k_vis = attn.k_proj(ln(vis_hs)).float()
                q_txt = attn.q_proj(ln(txt_hs)).float()
            scores = qvid_scores(q_txt, k_vis, num_heads, num_components)
            keep = select_topk(scores, num_tokens)

            keep_pos = torch.cat([
                torch.arange(vis_start, device=hs.device),
                vis_start + keep.to(hs.device),
                torch.arange(vis_end, hs.shape[1], device=hs.device),
            ])
            kwargs["attention_mask"] = None  # SDPA falls back to is_causal=True
            if kwargs.get("position_ids") is not None:
                kwargs["position_ids"] = kwargs["position_ids"][:, keep_pos]
                pruned_pos[0] = kwargs["position_ids"]
            _patch_rotary()
            return (hs[:, keep_pos, :],) + tuple(args[1:]), kwargs

        def _use_pruned_inputs(module, args, kwargs):
            # LlamaModel passes the same full-length mask / position ids to every
            # layer; swap in the pruned versions for the prefill pass.
            if args[0].shape[1] <= 1:
                return None
            kwargs["attention_mask"] = None
            if pruned_pos[0] is not None and kwargs.get("position_ids") is not None:
                kwargs["position_ids"] = pruned_pos[0]
            return args, kwargs

        state.handles.append(layers[scoring_layer].register_forward_pre_hook(
            _score_and_prune, with_kwargs=True))
        for layer in layers[scoring_layer + 1:]:
            state.handles.append(layer.register_forward_pre_hook(
                _use_pruned_inputs, with_kwargs=True))
        return result

    def _generate(self, *args, **kwargs):
        try:
            return original_generate(*args, **kwargs)
        finally:
            state.reset()

    model.prepare_inputs_labels_for_multimodal = types.MethodType(_pruned_prepare, model)
    model.generate = types.MethodType(_generate, model)
    return model
