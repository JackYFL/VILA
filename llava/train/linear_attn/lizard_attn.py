from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from llava.train.linear_attn.parallel_adapter import ParallelLinearAdapter


# ---------------------------------------------------------------------------
# Utility functions — copied from gated_lizard.py
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int, head_first: bool = True) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim).
    """
    if head_first:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    else:
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def vanilla_rope_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    dropout: float = 0.0,
):
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

    attn_output = flash_attn_func(
        q=query_states,
        k=key_states,
        v=value_states,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )
    return attn_output


def qwen2_flash_attention(
    query_states: torch.Tensor,       # (B, L, H_q, D) seq-first
    key_states: torch.Tensor,         # (B, L, H_k, D) seq-first
    value_states: torch.Tensor,       # (B, L, H_k, D) seq-first
    scaling: float,
    num_key_value_groups: int,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    dropout: float = 0.0,
) -> torch.Tensor:
    """
    Softmax attention following Qwen2FlashAttention2:
      1. Transpose to head-first (B, H, L, D)
      2. Apply RoPE with unsqueeze_dim=1 (head-first convention)
      3. repeat_kv for GQA
      4. Transpose back to seq-first (B, L, H, D) for flash_attn_func
      5. Call flash_attn_func with causal=True
    Returns attn_output in seq-first (B, L, H, D).
    """
    # Transpose to head-first for RoPE (Qwen2 convention)
    query_states = query_states.transpose(1, 2)   # (B, H_q, L, D)
    key_states   = key_states.transpose(1, 2)     # (B, H_k, L, D)
    value_states = value_states.transpose(1, 2)   # (B, H_k, L, D)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    # Expand k/v heads to match q heads (GQA)
    key_states   = repeat_kv(key_states,   num_key_value_groups, head_first=True)
    value_states = repeat_kv(value_states, num_key_value_groups, head_first=True)

    # Transpose back to seq-first for flash_attn (B, L, H, D)
    query_states = query_states.transpose(1, 2)
    key_states   = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Handle bf16/fp16 casting (mirrors Qwen2FlashAttention2)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        target_dtype = torch.bfloat16
        query_states = query_states.to(target_dtype)
        key_states   = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = flash_attn_func(
        q=query_states,
        k=key_states,
        v=value_states,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )

    return attn_output.to(input_dtype)


def sliding_sink_mask_bool(
    seq_len: int,
    window_size: int,
    sink_size: int,
    device=None,
) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)

    sliding = (col <= row) & (col >= row - (window_size - 1))
    sink = (col < sink_size) & (col <= row)

    return sliding | sink


def linear_fused_gated_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    log_gates: torch.Tensor,
    num_key_value_groups: int,
    dropout: float,
):
    key_states = repeat_kv(key_states, num_key_value_groups, head_first=False)
    value_states = repeat_kv(value_states, num_key_value_groups, head_first=False)
    log_gates = repeat_kv(log_gates[:, :, :, None], num_key_value_groups, head_first=False).squeeze(-1)

    attn_output, _ = chunk_simple_gla(
        q=query_states,
        k=key_states,
        v=value_states,
        g=log_gates,
        scale=1.0,
        output_final_state=False,
    )

    return attn_output


def linear_fused_gated_attention_func_with_cache(
    query_states: torch.Tensor,         # (B, L, H_q, K_feat)  seq-first
    key_states: torch.Tensor,           # (B, L, H_k, K_feat)  seq-first
    value_states: torch.Tensor,         # (B, L, H_k, V)       seq-first
    log_gates: torch.Tensor,            # (B, L, H_k)
    num_key_value_groups: int,
    past_hidden_state: Optional[torch.Tensor],  # (B, H_q, K_feat, V) or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cache-aware GLA forward.  Dispatches to:
      - fused_recurrent_simple_gla  for single-token generation (L == 1)
      - chunk_simple_gla            for multi-token prefill     (L  > 1)
    Both called with output_final_state=True to capture h_final for caching.

    Returns:
        attn_output:  (B, L, H_q, V)       seq-first
        final_state:  (B, H_q, K_feat, V)
    """
    # GQA expansion
    key_states   = repeat_kv(key_states,   num_key_value_groups, head_first=False)
    value_states = repeat_kv(value_states, num_key_value_groups, head_first=False)
    log_gates    = repeat_kv(
        log_gates[:, :, :, None], num_key_value_groups, head_first=False
    ).squeeze(-1)

    seq_len = query_states.shape[1]

    if seq_len == 1:
        attn_output, final_state = fused_recurrent_simple_gla(
            q=query_states,
            k=key_states,
            v=value_states,
            g=log_gates,
            scale=1.0,
            initial_state=past_hidden_state,
            output_final_state=True,
        )
    else:
        attn_output, final_state = chunk_simple_gla(
            q=query_states,
            k=key_states,
            v=value_states,
            g=log_gates,
            scale=1.0,
            initial_state=past_hidden_state,
            output_final_state=True,
        )

    return attn_output, final_state


# ---------------------------------------------------------------------------
# Feature map — copied from feature_maps.py
# ---------------------------------------------------------------------------

class MLP(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        feature_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim

        self.layer = nn.Parameter(
            torch.zeros((self.num_heads, self.head_dim, self.feature_dim),
                        dtype=dtype, device=device),
        )
        nn.init.kaiming_uniform_(self.layer)

    def forward(self, x: torch.Tensor, head_first: bool = True):
        if head_first:
            return torch.einsum("hdf,bhld->bhlf", self.layer, x)
        else:
            return torch.einsum("hdf,blhd->blhf", self.layer, x)


class FeatureMap(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        feature_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        eps: float = 1e-12,
        head_first: bool = True,
    ):
        super().__init__()
        self.mlp = MLP(
            num_heads=num_heads,
            head_dim=head_dim,
            feature_dim=feature_dim,
            dtype=dtype,
            device=device,
        )
        self.eps = eps
        self.head_first = head_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x, head_first=self.head_first)
        return torch.cat([
            torch.softmax(x, dim=-1),
            torch.softmax(-x, dim=-1),
        ], dim=-1).clamp(min=self.eps)


# ---------------------------------------------------------------------------
# LizardAttention — monkey-patch replacement for self_attn
# ---------------------------------------------------------------------------

class LizardAttention(nn.Module):
    """
    Monkey-patch replacement for self_attn during stage1 training.
    Mirrors GatedStreamingLinearAttention from gated_lizard.py but uses
    ParallelLinearAdapter for q/k/v instead of copying them directly.

    Trainable parameters: q/k/v adapter weights, feature_map_q/k, gated_proj.
    Frozen parameters: o_proj, rotary_emb.
    """

    def __init__(
        self,
        base_attention_module: nn.Module,
        use_base_attention: bool = False,
    ) -> None:
        super().__init__()

        # Copy config attributes (mirrors gated_lizard pattern)
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

        # Determine dtype/device from the original q_proj before creating adapters
        ref_dtype = base_attention_module.q_proj.weight.dtype
        ref_device = base_attention_module.q_proj.weight.device

        # Wrap q/k/v with ParallelLinearAdapter (trainable copies, initialized from originals)
        # self.q_proj_linear = ParallelLinearAdapter(base_attention_module.q_proj)
        # self.k_proj_linear = ParallelLinearAdapter(base_attention_module.k_proj)
        # self.v_proj_linear = ParallelLinearAdapter(base_attention_module.v_proj)
        self.q_proj = base_attention_module.q_proj
        self.k_proj = base_attention_module.k_proj
        self.v_proj = base_attention_module.v_proj
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            for p in proj.parameters():
                p.requires_grad_(False)
        

        # Keep o_proj frozen
        self.o_proj = base_attention_module.o_proj
        for p in self.o_proj.parameters():
            p.requires_grad_(False)

        # Keep rotary_emb frozen
        if hasattr(base_attention_module, "rotary_emb"):
            self.rotary_emb = base_attention_module.rotary_emb
            for p in self.rotary_emb.parameters():
                p.requires_grad_(False)

        # New trainable: feature maps for linear attention (seq-first, head_first=False)
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

        # New trainable: gate projection
        self.gated_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads,
            bias=True,
            dtype=ref_dtype,
            device=ref_device,
        )

        del base_attention_module  # free reference (matches gated_lizard pattern)

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
        # Gate for linear attention
        gates = self.gated_proj(hidden_states)              # (B, L, num_kv_heads)
        log_gates = F.logsigmoid(gates.float())             # fp32 for numerical stability

        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)   # (B, L, H_q, D)
        key_states   = self.k_proj(hidden_states).view(hidden_shape)   # (B, L, H_k, D)
        value_states = self.v_proj(hidden_states).view(hidden_shape)   # (B, L, H_k, D)

        # Linear gated attention — branch on cache presence
        # q/k/v are in seq-first format (B,L,H,D); feature_map uses head_first=False
        from llava.train.linear_attn.lizard_cache import LinearGatedCache

        if isinstance(past_key_value, LinearGatedCache):
            # Inference path: cache-aware recurrent kernel
            past_h = past_key_value.get_hidden_state(self.layer_idx)
            linear_attn_output, final_h = linear_fused_gated_attention_func_with_cache(
                query_states=self.feature_map_q(query_states),
                key_states=self.feature_map_k(key_states),
                value_states=value_states,
                log_gates=log_gates,
                num_key_value_groups=self.num_key_value_groups,
                past_hidden_state=past_h,
            )
            past_key_value.update_hidden_state(final_h, self.layer_idx, seq_len)
        else:
            # Training path: original behavior, no cache
            linear_attn_output = linear_fused_gated_attention_func(
                query_states=self.feature_map_q(query_states),
                key_states=self.feature_map_k(key_states),
                value_states=value_states,
                log_gates=log_gates,
                num_key_value_groups=self.num_key_value_groups,
                dropout=0.0 if not self.training else self.attention_dropout,
            )

        # Optionally compute vanilla softmax attention (use_base_attention=True path).
        # vanilla_rope_attention uses unsqueeze_dim=2 (seq-first), consistent with
        # gated_lizard. Standard Qwen2 uses unsqueeze_dim=1 (head-first), but since
        # our tensors are seq-first throughout, unsqueeze_dim=2 is correct here.
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

        # Apply RoPE for sink attention (seq-first, unsqueeze_dim=2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )

        # Sliding-window sink attention (transpose to head-first for SDPA)
        # sink_attn_output = F.scaled_dot_product_attention(
        #     query=query_states.transpose(1, 2),    # (B, H, L, D)
        #     key=key_states.transpose(1, 2),
        #     value=value_states.transpose(1, 2),
        #     dropout_p=0.0 if not self.training else self.attention_dropout,
        #     attn_mask=sliding_sink_mask_bool(seq_len, 128, 4, device=query_states.device),
        #     enable_gqa=True,
        # ).transpose(1, 2).contiguous()              # back to (B, L, H, D)

        # Combine linear + sink 50/50 (matches gated_lizard)
        # linear_attn_output = 0.5 * linear_attn_output + 0.5 * sink_attn_output
        linear_attn_output = linear_attn_output
        # import ipdb; ipdb.set_trace()
        # Choose output branch (mirrors gated_lizard output logic exactly)
        if self.use_base_attention:
            attn_output = softmax_attn_output.reshape(batch_size, seq_len, -1).contiguous()
        else:
            attn_output = linear_attn_output.reshape(batch_size, seq_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        # Expose cache for inference; None during training.
        # During inference the distillation tuple is not needed — return None to
        # avoid keeping linear_attn_output alive through the return chain.
        present_key_value = (
            past_key_value if isinstance(past_key_value, LinearGatedCache) else None
        )
        if present_key_value is not None:
            return attn_output, None, present_key_value
        return attn_output, (linear_attn_output, softmax_attn_output), present_key_value
