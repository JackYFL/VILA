from typing import List, Optional

import torch
from transformers.cache_utils import Cache


class LinearGatedCache(Cache):
    """
    Recurrent hidden-state cache for LizardAttention's GLA path.

    Stores the GLA hidden state h per layer, shape (B, H_q, K_feat, V) where
    K_feat = 2 * head_dim (doubled by FeatureMap's softmax-concat).

    GLA recurrence:  h_t = g_t * h_{t-1} + k_t^T @ v_t
    GLA output:      y_t = q_t @ h_t

    Inspired by LinearGatedDynamicCache from
    Lizard-main/dora_llm/models/modules/naive_cache_llama.py.
    Used during inference only; training code passes past_key_value=None.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens: int = 0
        # hidden_states[layer_idx]: (B, H_q, K_feat, V) or None
        self.hidden_states: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Cache protocol (required by transformers)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, layer_idx: int) -> Optional[torch.Tensor]:
        if layer_idx >= len(self.hidden_states):
            raise KeyError(
                f"Layer index {layer_idx} not found in cache "
                f"(cache has {len(self.hidden_states)} layers)."
            )
        return self.hidden_states[layer_idx]

    def __iter__(self):
        yield from self.hidden_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Number of tokens seen so far (used by Qwen2 for position offsets)."""
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None  # unbounded recurrent cache

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    # ------------------------------------------------------------------
    # GLA-specific accessors
    # ------------------------------------------------------------------

    def get_hidden_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return stored h for layer, or None if not yet populated (first prefill)."""
        if layer_idx >= len(self.hidden_states):
            return None
        return self.hidden_states[layer_idx]

    def update_hidden_state(
        self,
        new_h: torch.Tensor,
        layer_idx: int,
        seq_len: int,
    ) -> None:
        """
        Store the final hidden state returned by chunk/fused_recurrent_simple_gla.

        Args:
            new_h:     (B, H_q, K_feat, V) — final_state from FLA kernel
            layer_idx: which decoder layer
            seq_len:   tokens in this forward pass (advances _seen_tokens at layer 0)
        """
        if layer_idx == 0:
            self._seen_tokens += seq_len

        # Extend list if needed (handles non-contiguous layer_idx)
        if layer_idx >= len(self.hidden_states):
            self.hidden_states.extend(
                [None] * (layer_idx - len(self.hidden_states) + 1)
            )
        self.hidden_states[layer_idx] = new_h
