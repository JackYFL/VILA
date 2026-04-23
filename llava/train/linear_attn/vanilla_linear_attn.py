"""
Vanilla (non-gated) linear attention using fla.ops.linear_attn.

Subclasses LizardAttention so the existing monkey patches in monkey_patch.py
(which check isinstance(self_attn, LizardAttention)) and LinearGatedCache
both work unchanged.  The only behavioural differences vs Lizard are:
  - No gated_proj module (no gate), so one fewer trainable module per layer.
  - forward() calls chunk_linear_attn / fused_recurrent_linear_attn (from fla)
    instead of the *_simple_gla variants, dropping the gate argument.
  - normalize=True in the chunk kernel divides the output by the cumulative
    key sum for numerical stability (standard vanilla LA).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fla.ops.linear_attn import chunk_linear_attn, fused_recurrent_linear_attn

from llava.train.linear_attn.lizard_attn import (
    FeatureMap,
    LizardAttention,
    qwen2_flash_attention,
    repeat_kv,
)
from llava.train.linear_attn.lizard_cache import LinearGatedCache


def linear_attention_func(
    query_states: torch.Tensor,      # (B, L, H_q, K_feat) seq-first
    key_states: torch.Tensor,        # (B, L, H_k, K_feat) seq-first
    value_states: torch.Tensor,      # (B, L, H_k, V)      seq-first
    num_key_value_groups: int,
) -> torch.Tensor:
    key_states = repeat_kv(key_states, num_key_value_groups, head_first=False)
    value_states = repeat_kv(value_states, num_key_value_groups, head_first=False)
    attn_output, _ = chunk_linear_attn(
        q=query_states,
        k=key_states,
        v=value_states,
        scale=1.0,
        output_final_state=False,
        normalize=True,
    )
    return attn_output


def linear_attention_func_with_cache(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    past_hidden_state: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key_states, num_key_value_groups, head_first=False)
    value_states = repeat_kv(value_states, num_key_value_groups, head_first=False)

    seq_len = query_states.shape[1]
    if seq_len == 1:
        attn_output, final_state = fused_recurrent_linear_attn(
            q=query_states,
            k=key_states,
            v=value_states,
            scale=1.0,
            initial_state=past_hidden_state,
            output_final_state=True,
            normalize=False,
        )
    else:
        attn_output, final_state = chunk_linear_attn(
            q=query_states,
            k=key_states,
            v=value_states,
            scale=1.0,
            initial_state=past_hidden_state,
            output_final_state=True,
            normalize=True,
        )
    return attn_output, final_state


class VanillaLinearAttention(LizardAttention):
    """Vanilla (non-gated) linear attention.  Subclass of LizardAttention so
    the existing decoder-layer monkey patch and LinearGatedCache apply as-is.
    """

    def __init__(
        self,
        base_attention_module: nn.Module,
        use_base_attention: bool = False,
    ) -> None:
        # Skip LizardAttention.__init__ (which would create gated_proj);
        # nn.Module.__init__ is the only base setup we need.
        nn.Module.__init__(self)

        # Config attributes (matches LizardAttention.__init__)
        self.config = base_attention_module.config
        self.layer_idx = base_attention_module.layer_idx
        self.hidden_size = base_attention_module.config.hidden_size
        self.num_attention_heads = base_attention_module.config.num_attention_heads
        self.num_key_value_heads = base_attention_module.config.num_key_value_heads
        self.head_dim = base_attention_module.head_dim
        self.num_key_value_groups = base_attention_module.num_key_value_groups
        self.attention_dropout = base_attention_module.attention_dropout
        self.scaling = getattr(base_attention_module, "scaling", base_attention_module.head_dim ** -0.5)
        self.use_base_attention = use_base_attention

        ref_dtype = base_attention_module.q_proj.weight.dtype
        ref_device = base_attention_module.q_proj.weight.device

        # Frozen q/k/v/o projections
        self.q_proj = base_attention_module.q_proj
        self.k_proj = base_attention_module.k_proj
        self.v_proj = base_attention_module.v_proj
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            for p in proj.parameters():
                p.requires_grad_(False)

        self.o_proj = base_attention_module.o_proj
        for p in self.o_proj.parameters():
            p.requires_grad_(False)

        if hasattr(base_attention_module, "rotary_emb"):
            self.rotary_emb = base_attention_module.rotary_emb
            for p in self.rotary_emb.parameters():
                p.requires_grad_(False)

        # Trainable feature maps (same as Lizard)
        self.feature_map_q = FeatureMap(
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            feature_dim=self.head_dim,
            dtype=ref_dtype,
            device=ref_device,
            head_first=False,
        )
        self.feature_map_k = FeatureMap(
            num_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            feature_dim=self.head_dim,
            dtype=ref_dtype,
            device=ref_device,
            head_first=False,
        )

        # NOTE: no gated_proj — vanilla linear attention has no gate.

        del base_attention_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if isinstance(past_key_value, LinearGatedCache):
            past_h = past_key_value.get_hidden_state(self.layer_idx)
            linear_attn_output, final_h = linear_attention_func_with_cache(
                query_states=self.feature_map_q(query_states),
                key_states=self.feature_map_k(key_states),
                value_states=value_states,
                num_key_value_groups=self.num_key_value_groups,
                past_hidden_state=past_h,
            )
            past_key_value.update_hidden_state(final_h, self.layer_idx, seq_len)
        else:
            linear_attn_output = linear_attention_func(
                query_states=self.feature_map_q(query_states),
                key_states=self.feature_map_k(key_states),
                value_states=value_states,
                num_key_value_groups=self.num_key_value_groups,
            )

        softmax_attn_output = None
        if self.training and (self.use_base_attention or output_attentions):
            softmax_attn_output = qwen2_flash_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                scaling=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
                position_embeddings=position_embeddings,
                dropout=0.0 if not self.training else self.attention_dropout,
            )

        if self.use_base_attention:
            attn_output = softmax_attn_output.reshape(batch_size, seq_len, -1).contiguous()
        else:
            attn_output = linear_attn_output.reshape(batch_size, seq_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        present_key_value = (
            past_key_value if isinstance(past_key_value, LinearGatedCache) else None
        )
        if present_key_value is not None:
            return attn_output, None, present_key_value
        return attn_output, (linear_attn_output, softmax_attn_output), present_key_value
