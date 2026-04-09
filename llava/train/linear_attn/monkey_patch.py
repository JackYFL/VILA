from typing import Optional

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .lizard_attn import LizardAttention
from .lizard_cache import LinearGatedCache

_original_causal_lm_forward = None  # saved before patching


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    attn_result = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )

    if isinstance(self.self_attn, LizardAttention):
        # LizardAttention returns:
        #   training:  (attn_output, (linear_out, softmax_out), None)
        #   inference: (attn_output, None,                      cache)
        hidden_states = attn_result[0]
        attn_info = attn_result[1]          # (linear_out, softmax_out) or None
        present_key_value = attn_result[2] if len(attn_result) >= 3 else None
        # Compute distill loss INSIDE the checkpointed region so gradient checkpointing
        # can recompute it with grad_fn during backward. Store as scalar tensor.
        if isinstance(attn_info, tuple):
            linear_out, softmax_out = attn_info
            if output_attentions and softmax_out is not None:
                self_attn_weights = F.l1_loss(linear_out.float(), softmax_out.float())
            else:
                self_attn_weights = None
        else:
            self_attn_weights = None
    else:
        # Standard Qwen2Attention returns (attn_output, attn_weights, present_key_value)
        hidden_states = attn_result[0]
        self_attn_weights = attn_result[1] if output_attentions else None
        present_key_value = attn_result[2] if use_cache else None

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def patched_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    num_logits_to_keep: int = 0,
    stage_type=None,
    **loss_kwargs,
):
    # Replace DynamicCache with LinearGatedCache for LizardAttention inference.
    # Qwen2Model.forward creates DynamicCache() when past_key_values is None and
    # use_cache=True.  Since LinearGatedCache extends Cache, injecting it here causes
    # Qwen2Model to skip that creation (guard: `not isinstance(past_key_values, Cache)`).
    _use_cache = use_cache if use_cache is not None else self.config.use_cache
    if _use_cache and past_key_values is None and not self.training:
        _inner = getattr(self.model, "model", self.model)
        _layers = getattr(_inner, "layers", None)
        if _layers and isinstance(getattr(_layers[0], "self_attn", None), LizardAttention):
            past_key_values = LinearGatedCache()

    # stage_type=None: delegate to original Qwen2ForCausalLM.forward unchanged
    if stage_type is None:
        return _original_causal_lm_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs,
        )

    # stage1: force output_attentions=True to expose per-layer linear+softmax attention
    if stage_type == "stage1":
        output_attentions = True

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        # stage1: attentions[i] = (linear_attn_output, softmax_attn_output) per layer
    )


def apply_linear_attn_monkey_patches():
    """Apply monkey patches to Qwen2DecoderLayer and Qwen2ForCausalLM.

    After calling this function:
    - Qwen2DecoderLayer.forward handles LizardAttention's 2-tuple return correctly.
    - Qwen2ForCausalLM.forward gains a `stage_type` parameter:
        - stage_type=None (default): original Qwen2 behavior, nothing changed.
        - stage_type="stage1": outputs per-layer (linear_attn_output, softmax_attn_output)
          in CausalLMOutputWithPast.attentions.
    """
    global _original_causal_lm_forward
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
    _original_causal_lm_forward = Qwen2ForCausalLM.forward
    Qwen2DecoderLayer.forward = patched_decoder_layer_forward
    Qwen2ForCausalLM.forward = patched_causal_lm_forward
