"""QViD: Vision Token Pruning via Query-Vision Interaction Decomposition.

Core implementation for LLaVA-1.5 (transformers 4.37.2).

`QViDLlamaModel` replaces `LlamaModel` as the language model of LLaVA
(see llava_llama.py). During the prefill pass, at the input of decoder layer
`qvid_layer` (default 3), it scores the visual tokens with that layer's own
query/key projections and keeps the `qvid_num_tokens` most query-relevant ones:

    layers 0 .. L-1   see all visual tokens (576 for LLaVA-1.5)
    layer  L          visual tokens scored and pruned
    layers L .. end   see only the kept tokens

Scoring, per attention head h, with Q = text-token queries and K = visual-token
keys of layer L:

    M_h = Q_h K_h^T                       (N_q x N_v affinity, N_q << N_v)
    M_h^T = Q R,  R^T = U S V^T            (reduced QR, then an N_q x N_q SVD)
    right singular vectors of M_h = Q V
    score_j = || row j of the top-r right singular vectors ||^2, averaged over heads

Kept tokens keep their original position ids, so RoPE is unchanged for them and
for every decoding step. Pruning is off until `set_qvid()` is called.
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel, LlamaRotaryEmbedding


def qvid_scores(q_txt: torch.Tensor, k_vis: torch.Tensor, num_heads: int,
                num_components: int = 16) -> torch.Tensor:
    """Per-head leverage scores of the text-visual affinity.

    Args:
        q_txt: (N_q, D) query projections of the text tokens.
        k_vis: (N_v, D) key projections of the visual tokens.
        num_heads: attention heads; D must be divisible by it.
        num_components: rank r of the singular subspace (clipped to N_q).

    Returns:
        (N_v,) float32 scores; higher means more relevant to the text.
    """
    n_q, dim = q_txt.shape
    n_v = k_vis.shape[0]
    head_dim = dim // num_heads

    q = q_txt.view(n_q, num_heads, head_dim).transpose(0, 1).float()  # (H, N_q, d)
    k = k_vis.view(n_v, num_heads, head_dim).transpose(0, 1).float()  # (H, N_v, d)
    m = q @ k.transpose(-1, -2)                                       # (H, N_q, N_v)
    r = min(num_components, n_q, n_v)

    q_qr, r_qr = torch.linalg.qr(m.mT)                                # (H, N_v, N_q), (H, N_q, N_q)
    _, _, vh = torch.linalg.svd(r_qr.mT, full_matrices=False)         # (H, N_q, N_q)
    w = (q_qr @ vh.mT)[:, :, :r]                                      # (H, N_v, r)

    return (w * w).sum(dim=-1).mean(dim=0)                            # (N_v,)


class FullTableRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding that returns its whole cos/sin table.

    The stock module returns the first `seq_len` (= KV length) rows. After
    pruning, a layer's KV length is shorter than the largest position id of the
    tokens it holds; `apply_rotary_pos_emb` indexes the table by position id, so
    the table must not be truncated. Values are identical to the stock module's.
    """

    def forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)

    def ensure_length(self, length: int, x: torch.Tensor):
        if length > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=length, device=x.device, dtype=x.dtype)


class QViDLlamaModel(LlamaModel):
    """LlamaModel with QViD visual-token pruning inside the prefill pass."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.qvid_num_tokens = None    # None = no pruning (vanilla LLaVA)
        self.qvid_layer = 3
        self.qvid_num_components = 16
        # (start, length) of the visual tokens in the current sequence; set by
        # LlavaLlamaForCausalLM for each new prompt and consumed by the prefill.
        self.qvid_vision_span = None

        if getattr(config, "rope_scaling", None) is not None:
            raise NotImplementedError("QViD supports rope_scaling=None only.")
        for layer in self.layers:
            attn = layer.self_attn
            attn.rotary_emb = FullTableRotaryEmbedding(
                attn.head_dim, max_position_embeddings=attn.max_position_embeddings, base=attn.rope_theta)

    def set_qvid(self, num_tokens: Optional[int] = 64, layer: int = 3, num_components: int = 16):
        """Enable QViD: keep `num_tokens` visual tokens from decoder layer `layer` on.

        num_tokens=None (or >= the number of visual tokens) disables pruning.
        """
        if num_tokens is not None:
            if not self._use_sdpa:
                raise ValueError("QViD requires attn_implementation='sdpa' (pruned layers rely "
                                 "on SDPA's implicit causal mask).")
            if self.config.num_key_value_heads != self.config.num_attention_heads:
                raise NotImplementedError("QViD scoring assumes num_key_value_heads == num_attention_heads.")
            if getattr(self.config, "image_aspect_ratio", "square") == "anyres" or \
                    getattr(self.config, "mm_patch_merge_type", "flat") != "flat":
                raise NotImplementedError(
                    "QViD supports LLaVA-1.5 only: this model uses a variable number of visual "
                    "tokens per image (LLaVA-1.6 'anyres' / non-flat patch merging).")
            if not 0 <= layer < len(self.layers):
                raise ValueError(f"layer must be in [0, {len(self.layers) - 1}], got {layer}")
            if num_tokens < 1:
                raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        self.qvid_num_tokens = num_tokens
        self.qvid_layer = layer
        self.qvid_num_components = num_components

    @torch.no_grad()
    def _qvid_prune(self, layer, hidden_states, position_ids, span):
        """Score visual tokens with `layer`'s Q/K projections and drop all but the top ones."""
        start, n_vis = span
        end = start + n_vis
        seq_len = hidden_states.shape[1]

        # Text = every token after the image (instruction and "ASSISTANT:").
        vis = hidden_states[0, start:end]
        txt = hidden_states[0, end:] if end < seq_len else hidden_states[0, :start]

        k_vis = layer.self_attn.k_proj(layer.input_layernorm(vis)).float()
        q_txt = layer.self_attn.q_proj(layer.input_layernorm(txt)).float()
        scores = qvid_scores(q_txt, k_vis, self.config.num_attention_heads, self.qvid_num_components)
        keep = scores.topk(self.qvid_num_tokens, largest=True).indices.sort().values

        device = hidden_states.device
        keep_idx = torch.cat([
            torch.arange(start, device=device),
            start + keep.to(device),
            torch.arange(end, seq_len, device=device),
        ])
        return hidden_states[:, keep_idx], position_ids[:, keep_idx]

    def _qvid_take_span(self, batch_size, seq_length, past_key_values_length, attention_mask):
        """Return the visual-token span to prune in this pass, or None.

        Each prefill consumes the span set for its prompt, whether or not it prunes.
        """
        if seq_length <= 1:
            return None  # decoding step
        span, self.qvid_vision_span = self.qvid_vision_span, None
        if self.qvid_num_tokens is None or span is None:
            return None
        start, n_vis = span
        if batch_size != 1 or past_key_values_length != 0:
            return None  # not a fresh single-sample prefill
        if self.qvid_num_tokens >= n_vis or start + n_vis > seq_length:
            return None
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError("QViD does not support padded inputs.")
        return span

    # Same as transformers 4.37.2 LlamaModel.forward, plus the QViD step marked below.
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # QViD: decide before the mask is converted to 4D.
        qvid_span = self._qvid_take_span(batch_size, seq_length, past_key_values_length, attention_mask)
        if self.qvid_num_tokens is not None:
            # Pruned layers hold fewer KV entries than the largest position id;
            # make sure every RoPE table covers the full (unpruned) length.
            for layer in self.layers:
                layer.self_attn.rotary_emb.ensure_length(past_key_values_length + seq_length, inputs_embeds)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            # ---- QViD: prune visual tokens at the input of layer `qvid_layer` ----
            if qvid_span is not None and layer_idx == self.qvid_layer:
                hidden_states, position_ids = self._qvid_prune(decoder_layer, hidden_states, position_ids, qvid_span)
                attention_mask = None  # unpadded single sample: SDPA uses is_causal on the new length
            # ----------------------------------------------------------------------

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
