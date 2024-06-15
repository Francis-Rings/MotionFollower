import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers
import xformers.ops
import numpy as np
import math
from einops import rearrange


class AttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        camera=False,
    ):
        super().__init__()
        self.start_point = 60
        if camera == "True":
            self.start_point = 20

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        iter_cur=None,
        save_kv=None,
        source_masks=None,
        target_masks=None,
        camera_movement=True,
        long_context=None,
        inference_num=50,
    ):
        start_point = self.start_point
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if encoder_hidden_states is not None:
            is_self_attention = False
        else:
            is_self_attention = True

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]

        query = attn.head_to_batch_dim(query)

        if attn.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.updown == 'up' and iter_cur >= start_point and is_self_attention and not save_kv and (long_context is not None):
            key_ref = torch.cat([attn.buffer_key[iter_cur][c] for c in long_context[0]], dim=0).to('cuda', dtype=query.dtype)
            value_ref = torch.cat([attn.buffer_value[iter_cur][c] for c in long_context[0]], dim=0).to('cuda', dtype=query.dtype)
            if camera_movement:
                target_width = math.sqrt(value_ref.size()[1])
                target_height = target_width
                source_masks = F.interpolate(source_masks, size=(int(target_height), int(target_width)), mode="nearest")
                background_source_masks = 1-source_masks
                background_source_masks = rearrange(background_source_masks, "f c h w -> f (h w) c")
                key_ref = key_ref * background_source_masks
                value_ref = value_ref * background_source_masks
                key_ref = key_ref.repeat(2, 1, 1)
                value_ref = value_ref.repeat(2, 1, 1)
                target_masks = F.interpolate(target_masks, size=(int(target_height), int(target_width)), mode="nearest")

                foreground_target_masks = rearrange(target_masks, "f c h w -> f (h w) c")
                foreground_target_masks = torch.cat([foreground_target_masks] * 2, dim=0)
                key = key * foreground_target_masks
                value = value * foreground_target_masks
                key = torch.cat([key, key_ref], dim=1)
                value = torch.cat([value, value_ref], dim=1)
            else:
                key_ref = key_ref.repeat(2,1,1)
                value_ref = value_ref.repeat(2, 1, 1)
                key = torch.cat([key, key_ref], dim=1)
                value = torch.cat([value, value_ref], dim=1)
        elif attn.updown == 'up' and iter_cur >= start_point and is_self_attention and not save_kv:
            key_ref = attn.buffer_key[iter_cur].to('cuda', dtype=query.dtype)
            value_ref = attn.buffer_value[iter_cur].to('cuda', dtype=query.dtype)

            if camera_movement:
                target_width = math.sqrt(value_ref.size()[1])
                target_height = target_width
                source_masks = F.interpolate(source_masks, size=(int(target_height), int(target_width)), mode="nearest")
                background_source_masks = 1-source_masks
                background_source_masks = rearrange(background_source_masks, "f c h w -> f (h w) c")
                key_ref = key_ref * background_source_masks
                value_ref = value_ref * background_source_masks
                key_ref = key_ref.repeat(2, 1, 1)
                value_ref = value_ref.repeat(2, 1, 1)
                target_masks = F.interpolate(target_masks, size=(int(target_height), int(target_width)), mode="nearest")

                foreground_target_masks = rearrange(target_masks, "f c h w -> f (h w) c")
                foreground_target_masks = torch.cat([foreground_target_masks] * 2, dim=0)
                key = key * foreground_target_masks
                value = value * foreground_target_masks
                key = torch.cat([key, key_ref], dim=1)
                value = torch.cat([value, value_ref], dim=1)
            else:
                key_ref = key_ref.repeat(2,1,1)
                value_ref = value_ref.repeat(2, 1, 1)
                key = torch.cat([key, key_ref], dim=1)
                value = torch.cat([value, value_ref], dim=1)


        if attn.updown == 'up' and save_kv and is_self_attention and (long_context is not None) and iter_cur >= start_point:
            if not hasattr(attn, 'buffer_key'):
                attn.buffer_key = {}
                attn.buffer_value = {}
                for i in range(start_point, inference_num):
                    attn.buffer_key[i] = {}
                    attn.buffer_value[i] = {}

            for idx, element in enumerate(long_context[0]):
                attn.buffer_key[iter_cur][element] = torch.unsqueeze(key[idx, :, :], dim=0).detach().cpu()
                attn.buffer_value[iter_cur][element] = torch.unsqueeze(value[idx, :, :], dim=0).detach().cpu()
        elif attn.updown == 'up' and save_kv and is_self_attention and iter_cur >= start_point:
            if not hasattr(attn, 'buffer_key'):
                attn.buffer_key = {}
                attn.buffer_value = {}
            attn.buffer_key[iter_cur] = key.cpu()
            attn.buffer_value[iter_cur] = value.cpu()


        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)


        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states