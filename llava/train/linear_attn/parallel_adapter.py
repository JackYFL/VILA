import copy
import torch
import torch.nn as nn
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


class ParallelLinearAdapter(nn.Module):
    """
    Creates a trainable copy of a linear projection, initialized by deep-copying
    the original layer's weights. Only self.adapter is stored; the original layer
    is used solely for weight initialization and discarded.

    Under DeepSpeed ZeRO-3, parameters are sharded across GPUs, so a naive
    deepcopy would only capture the local shard. GatheredParameters is used
    to temporarily gather the full weights on all ranks before copying.

    Only self.adapter parameters have requires_grad=True.
    """

    def __init__(self, original: nn.Module) -> None:
        super().__init__()

        if not hasattr(original, "in_features") or not hasattr(original, "out_features"):
            raise TypeError(
                f"ParallelLinearAdapter requires a module with .in_features and .out_features, "
                f"got {type(original).__name__}"
            )

        has_bias = original.bias is not None

        if is_deepspeed_zero3_enabled():
            import deepspeed
            # Under ZeRO-3, parameters are sharded — gather full weights before copying.
            # modifier_rank=None means read-only gather on all ranks.
            params = list(original.parameters())
            ref = params[0]
            self.adapter = nn.Linear(
                original.in_features, original.out_features,
                bias=has_bias, device=ref.device, dtype=ref.dtype,
            )
            with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
                with torch.no_grad():
                    self.adapter.weight.data.copy_(original.weight.data)
                    if has_bias:
                        self.adapter.bias.data.copy_(original.bias.data)
        else:
            self.adapter = copy.deepcopy(original)

        for p in self.adapter.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

    def __repr__(self) -> str:
        return (
            f"ParallelLinearAdapter("
            f"in_features={self.adapter.in_features}, "
            f"out_features={self.adapter.out_features}, "
            f"bias={self.adapter.bias is not None})"
        )
