"""Local Qwen3.5 compatibility layer for older transformers builds."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
    Qwen3NextRMSNorm,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, check_model_inputs


class Qwen3_5TextConfig(Qwen3NextConfig):
    model_type = "qwen3_5_text"

    def __init__(self, rope_parameters=None, rope_scaling=None, **kwargs):
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters
        super().__init__(rope_scaling=rope_scaling, **kwargs)


class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_5"


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


class Qwen3_5TextModel(Qwen3NextModel):
    """Qwen3.5 text backbone: Qwen3-Next layers with multimodal RoPE."""

    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        Qwen3NextPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3NextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        patched = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = patched
        return hidden_states

    def _normalize_attention_mask(self, attention_mask: Optional[torch.Tensor | dict[str, Any]]) -> Optional[torch.Tensor]:
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask.get("full_attention")
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
            if attention_mask.dtype.is_floating_point:
                attention_mask = attention_mask / torch.finfo(attention_mask.dtype).min
                attention_mask = (1.0 - attention_mask).int()
        return attention_mask

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor | dict[str, Any]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else None

        attention_mask_tensor = self._normalize_attention_mask(attention_mask)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask_tensor,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask_tensor, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if deepstack_visual_embeds is not None and visual_pos_masks is not None:
                if layer_idx < len(deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                    )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Qwen3_5Model(Qwen3VLModel):
    config: Qwen3_5Config
    _no_split_modules = ["Qwen3NextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config: Qwen3_5Config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = Qwen3_5Config

    def __init__(self, config: Qwen3_5Config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

