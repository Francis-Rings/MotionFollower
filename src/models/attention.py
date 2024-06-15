from typing import Any, Dict, Optional, Callable

import torch
from diffusers.models.attention import AdaLayerNorm, FeedForward
from einops import rearrange
from torch import nn
from diffusers.utils.import_utils import is_xformers_available

from src.models.attn_process_diffuser import Attention

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    print(1/0)
    xformers = None

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TemporalBasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
            unet_use_cross_frame_attention=None,
            unet_use_temporal_attention=None,
            updown=None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        # SC-Attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            updown=updown,
        )
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False

        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )

    def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            attention_mask=None,
            video_length=None,
            iter_cur=None,
            save_kv=None,
            source_masks=None,
            target_masks=None,
            long_context=None,
    ):

        norm_hidden_states = (
            self.norm1(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm1(hidden_states)
        )
        if self.unet_use_cross_frame_attention:
            hidden_states = (
                self.attn1(
                    norm_hidden_states,
                    attention_mask=attention_mask,
                    video_length=video_length,
                    iter_cur=iter_cur,
                    save_kv=save_kv,
                    source_masks=source_masks,
                    target_masks=target_masks,
                    long_context=long_context,
                )
                + hidden_states
            )
        else:
            if iter_cur is not None and save_kv is not None:
                hidden_states = (
                    self.attn1(norm_hidden_states, attention_mask=attention_mask, iter_cur=iter_cur,save_kv=save_kv,source_masks=source_masks, target_masks=target_masks, long_context=long_context)
                    + hidden_states
                )
            else:
                hidden_states = (
                        self.attn1(norm_hidden_states, attention_mask=attention_mask,source_masks=source_masks, target_masks=target_masks,)+hidden_states
                )

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                    self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                    )
                    + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
