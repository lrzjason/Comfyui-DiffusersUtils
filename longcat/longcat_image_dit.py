# Copied from https://github.com/meituan-longcat/LongCat-Image

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_flux import \
    FluxTransformerBlock, FluxSingleTransformerBlock, \
    AdaLayerNormContinuous, Transformer2DModelOutput
from diffusers.models.embeddings import Timesteps, TimestepEmbedding,FluxPosEmbed
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from accelerate.logging import get_logger
from diffusers.loaders import PeftAdapterMixin

logger = get_logger(__name__, log_level="INFO")


class TimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        return timesteps_emb
    

class LongCatImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin ):
    """
    The Transformer model introduced in Flux.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 3584,
        axes_dims_rope: List[int] = [16, 56, 56],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pooled_projection_dim = pooled_projection_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_embed = TimestepEmbeddings(embedding_dim=self.inner_dim)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.initialize_weights()

        self.use_checkpoint = [True] * num_layers
        self.use_single_checkpoint = [True] * num_single_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The  forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = self.time_embed( timestep, hidden_states.dtype )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_checkpoint[index_block]:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_single_checkpoint[index_block]:
                encoder_hidden_states,hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.context_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm1_context.linear.weight, 0)
            nn.init.constant_(block.norm1_context.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.norm.linear.weight, 0)
            nn.init.constant_(block.norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)


# class BlockSwapLongCatImageTransformer2DModel(LongCatImageTransformer2DModel):
#     """
#     Transformer model for flow matching on sequences.
#     """

#     @register_to_config
#     def __init__(
#         self,
#         **kwargs
#     ):
#         super().__init__(
#             **kwargs
#         )
        
#         self.cpu_offload_checkpointing = False
#         self.blocks_to_swap = None

#         self.offloader_double = None
#         self.offloader_single = None
#         self.num_double_blocks = len(self.transformer_blocks)
#         self.num_single_blocks = len(self.single_transformer_blocks)
        
        
#     def enable_block_swap(self, num_blocks: int, device: torch.device):
#         self.blocks_to_swap = num_blocks
#         double_blocks_to_swap = num_blocks // 2
#         single_blocks_to_swap = (num_blocks - double_blocks_to_swap) * 2

#         assert double_blocks_to_swap <= self.num_double_blocks - 2 and single_blocks_to_swap <= self.num_single_blocks - 2, (
#             f"Cannot swap more than {self.num_double_blocks - 2} double blocks and {self.num_single_blocks - 2} single blocks. "
#             f"Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks."
#         )

#         self.offloader_double = custom_offloading_utils.ModelOffloader(
#             self.transformer_blocks, self.num_double_blocks, double_blocks_to_swap, device  # , debug=True
#         )
#         self.offloader_single = custom_offloading_utils.ModelOffloader(
#             self.single_transformer_blocks, self.num_single_blocks, single_blocks_to_swap, device  # , debug=True
#         )
#         print(
#             f"FLUX: Block swap enabled. Swapping {num_blocks} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}."
#         )
#     def move_to_device_except_swap_blocks(self, device: torch.device):
#         # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
#         if self.blocks_to_swap:
#             save_transformer_blocks = self.transformer_blocks
#             save_single_transformer_blocks = self.single_transformer_blocks
#             self.transformer_blocks = None
#             self.single_transformer_blocks = None

#         self.to(device)

#         if self.blocks_to_swap:
#             self.transformer_blocks = save_transformer_blocks
#             self.single_transformer_blocks = save_single_transformer_blocks

#     def prepare_block_swap_before_forward(self):
#         if self.blocks_to_swap is None or self.blocks_to_swap == 0:
#             return
#         self.offloader_double.prepare_block_devices_before_forward(self.transformer_blocks)
#         self.offloader_single.prepare_block_devices_before_forward(self.single_transformer_blocks)


#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: torch.Tensor = None,
#         timestep: torch.LongTensor = None,
#         img_ids: torch.Tensor = None,
#         txt_ids: torch.Tensor = None,
#         guidance: torch.Tensor = None,
#         return_dict: bool = True,
#     ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
#         """
#         The  forward method.

#         Args:
#             hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
#                 Input `hidden_states`.
#             encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
#                 Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
#             timestep ( `torch.LongTensor`):
#                 Used to indicate denoising step.
#             block_controlnet_hidden_states: (`list` of `torch.Tensor`):
#                 A list of tensors that if specified are added to the residuals of transformer blocks.
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
#                 tuple.

#         Returns:
#             If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
#             `tuple` where the first element is the sample tensor.
#         """
#         hidden_states = self.x_embedder(hidden_states)

#         timestep = timestep.to(hidden_states.dtype) * 1000
#         if guidance is not None:
#             guidance = guidance.to(hidden_states.dtype) * 1000
#         else:
#             guidance = None

#         temb = self.time_embed( timestep, hidden_states.dtype )
#         encoder_hidden_states = self.context_embedder(encoder_hidden_states)

#         if txt_ids.ndim == 3:
#             logger.warning(
#                 "Passing `txt_ids` 3d torch.Tensor is deprecated."
#                 "Please remove the batch dimension and pass it as a 2d torch Tensor"
#             )
#             txt_ids = txt_ids[0]
#         if img_ids.ndim == 3:
#             logger.warning(
#                 "Passing `img_ids` 3d torch.Tensor is deprecated."
#                 "Please remove the batch dimension and pass it as a 2d torch Tensor"
#             )
#             img_ids = img_ids[0]

#         ids = torch.cat((txt_ids, img_ids), dim=0)
#         image_rotary_emb = self.pos_embed(ids)

#         for index_block, block in enumerate(self.transformer_blocks):
#             if self.blocks_to_swap:
#                 self.offloader_double.wait_for_block(index_block)
            
#             if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_checkpoint[index_block]:
#                 encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
#                     block,
#                     hidden_states,
#                     encoder_hidden_states,
#                     temb,
#                     image_rotary_emb,
#                 )
#             else:
#                 encoder_hidden_states, hidden_states = block(
#                     hidden_states=hidden_states,
#                     encoder_hidden_states=encoder_hidden_states,
#                     temb=temb,
#                     image_rotary_emb=image_rotary_emb,
#                 )

#         for index_block, block in enumerate(self.single_transformer_blocks):
#             if self.blocks_to_swap:
#                 self.offloader_single.wait_for_block(index_block)
                
#             if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_single_checkpoint[index_block]:
#                 encoder_hidden_states,hidden_states = self._gradient_checkpointing_func(
#                     block,
#                     hidden_states,
#                     encoder_hidden_states,
#                     temb,
#                     image_rotary_emb,
#                 )
#             else:
#                 encoder_hidden_states, hidden_states = block(
#                     hidden_states=hidden_states,
#                     encoder_hidden_states=encoder_hidden_states,
#                     temb=temb,
#                     image_rotary_emb=image_rotary_emb,
#                 )

#         hidden_states = self.norm_out(hidden_states, temb)
#         output = self.proj_out(hidden_states)

#         if not return_dict:
#             return (output,)

#         return Transformer2DModelOutput(sample=output)