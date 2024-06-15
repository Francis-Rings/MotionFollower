import copy
import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor
import torch.nn.functional as F
from src.pipelines.utils import get_tensor_interpolation_method


@dataclass
class MotionEditorOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class MotionEditor(DiffusionPipeline):
    _optional_components = []

    def __init__(
            self,
            vae,
            image_encoder,
            denoising_unet,
            pose_guider,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            image_proj_model=None,
            tokenizer=None,
            text_encoder=None,
            estimator=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            estimator=estimator,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx: frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            width,
            height,
            video_length,
            dtype,
            device,
            generator,
            latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )

        if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
            self,
            ref_images,
            target_pose_images,
            width,
            height,
            video_length,
            num_inference_steps,
            guidance_scale,
            num_images_per_prompt=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            context_schedule="uniform",
            context_frames=24,
            context_stride=1,
            context_overlap=4,
            context_batch_size=1,
            interpolation_factor=1,
            SDE_strength=0.4,
            SDE_strength_un=0,
            energy_scale=4.0,
            start_time=50,
            source_masks=None,
            target_masks=None,
            ref_pose_images=None,
            ref_latents_collections=None,
            latents_org=None,
            save_kv=None,
            pose_scale=1.0,
            **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_images_list = []
        for ref_clip_image in ref_images:
            clip_image = self.clip_image_processor.preprocess(
                ref_clip_image.resize((224, 224)), return_tensors="pt"
            ).pixel_values
            clip_images_list.append(clip_image)
        clip_images = torch.stack(clip_images_list, dim=0)
        clip_images = clip_images.squeeze()
        clip_image_embeds = self.image_encoder(
            clip_images.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states_cfg = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_images_list = []
        for ref_image in ref_images:
            ref_image_tensor = self.ref_image_processor.preprocess(
                ref_image, height=height, width=width
            )  # (bs, c, width, height)
            ref_images_list.append(ref_image_tensor)
        ref_images_tensor = torch.stack(ref_images_list, dim=0)
        ref_images_tensor = ref_images_tensor.squeeze()
        ref_images_tensor = ref_images_tensor.unsqueeze(0)
        appearance_tensor = ref_images_tensor.to(dtype=self.denoising_unet.dtype, device=self.denoising_unet.device)

        latents = latents_org

        # Prepare a list of pose condition images
        target_pose_cond_tensor_list = []
        for target_pose_image in target_pose_images:
            target_pose_cond_tensor = self.cond_image_processor.preprocess(target_pose_image, height=height, width=width)
            target_pose_cond_tensor = target_pose_cond_tensor.squeeze().unsqueeze(1)
            target_pose_cond_tensor_list.append(target_pose_cond_tensor)

        target_pose_cond_tensor = torch.cat(target_pose_cond_tensor_list, dim=1)  # (c, t, h, w)
        target_pose_cond_tensor = target_pose_cond_tensor.unsqueeze(0)
        target_pose_cond_tensor = target_pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        target_pose_fea = self.pose_guider(target_pose_cond_tensor)
        target_pose_fea_cfg = (
            torch.cat([target_pose_fea] * 2) if do_classifier_free_guidance else target_pose_fea
        )

        ref_pose_cond_tensor_list = []
        for ref_pose_image in ref_pose_images:
            ref_pose_cond_tensor = self.cond_image_processor.preprocess(ref_pose_image, height=height,
                                                                           width=width)
            ref_pose_cond_tensor = ref_pose_cond_tensor.squeeze().unsqueeze(1)
            ref_pose_cond_tensor_list.append(ref_pose_cond_tensor)

        ref_pose_cond_tensor = torch.cat(ref_pose_cond_tensor_list, dim=1)  # (c, t, h, w)
        ref_pose_cond_tensor = ref_pose_cond_tensor.unsqueeze(0)
        ref_pose_cond_tensor = ref_pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        ref_pose_fea = self.pose_guider(ref_pose_cond_tensor)
        ref_pose_fea_cfg = (
            torch.cat([ref_pose_fea] * 2) if do_classifier_free_guidance else ref_pose_fea
        )

        source_masks = source_masks.to(dtype=self.denoising_unet.dtype, device=self.denoising_unet.device)  # [24, 1, 512, 512]
        target_masks = target_masks.to(dtype=self.denoising_unet.dtype, device=self.denoising_unet.device)  # [24, 1, 512, 512]
        energy_scale = energy_scale * 1e3

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
                next_timestep = max(next_timestep, 0)
                if energy_scale == 0:
                    replay = 1
                elif 20 < i < 30 and i % 2 == 0:
                    replay = 3
                else:
                    replay = 1
                for ri in range(replay):
                    latent_model_input = (
                        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    with torch.no_grad():
                        noise_pred = self.denoising_unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=encoder_hidden_states_cfg,
                            pose_cond_fea=target_pose_fea_cfg,
                            return_dict=False,
                            ref_images=appearance_tensor,
                            save_kv=save_kv,
                            iter_cur=i,
                            source_masks=source_masks,
                            target_masks=target_masks,
                            pose_scale=pose_scale,
                        )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                        )

                    if energy_scale != 0 and i < 30 and (i % 2 == 0 or i < 10):
                        noise_pred_org = noise_pred
                        guidance = self.guidance_edit(latent=latents,
                                                      latent_noise_ref=ref_latents_collections[-(i + 1)],
                                                      t=t,
                                                      energy_scale=energy_scale,
                                                      up_ft_index=[1, 2],
                                                      source_masks=source_masks,
                                                      target_masks=target_masks,
                                                      target_pose_fea=target_pose_fea,
                                                      ref_pose_fea=ref_pose_fea,
                                                      encoder_hidden_states=encoder_hidden_states,
                                                      appearance_tensor=appearance_tensor
                                                      )
                        noise_pred = noise_pred + guidance
                    else:
                        noise_pred_org = None

                    prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                    beta_prod_t = 1 - alpha_prod_t

                    pred_original_sample = (alpha_prod_t ** 0.5) * latents - (beta_prod_t ** 0.5) * noise_pred
                    pred_epsilon = (alpha_prod_t ** 0.5) * noise_pred + (beta_prod_t ** 0.5) * latents

                    if 10 < i < 20:
                        eta, eta_rd = SDE_strength_un, SDE_strength
                    else:
                        eta, eta_rd = 0., 0.

                    variance = self.scheduler._get_variance(t, prev_timestep)
                    std_dev_t = eta * variance ** (0.5)
                    std_dev_t_rd = eta_rd * variance ** (0.5)

                    if noise_pred_org is not None:
                        pred_epsilon_org = (alpha_prod_t ** 0.5) * noise_pred_org + (beta_prod_t ** 0.5) * latents
                        pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd ** 2) ** (0.5) * pred_epsilon_org
                        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon_org
                    else:
                        pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd ** 2) ** (0.5) * pred_epsilon
                        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon
                    latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                    latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd

                    if eta_rd > 0 or eta>0:
                        variance_noise = randn_tensor(noise_pred.shape, generator=generator, device=noise_pred.device, dtype=noise_pred.dtype)
                        variance_rd = std_dev_t_rd * variance_noise
                        variance = std_dev_t * variance_noise
                        masks = target_masks.unsqueeze(0)
                        masks = rearrange(masks, "b f c h w -> b c f h w")
                        masks = F.interpolate(masks, size=latent_prev.size()[-3:], mode="nearest")
                        latent_prev = (latent_prev + variance) * (1 - masks) + (latent_prev_rd + variance_rd) * masks

                    if replay > 1:
                        with torch.no_grad():
                            alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                            alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                            beta_prod_t = 1 - alpha_prod_t
                            model_output = self.denoising_unet(
                                latent_prev,
                                next_timestep,
                                encoder_hidden_states=encoder_hidden_states,
                                pose_cond_fea=target_pose_fea,
                                return_dict=False,
                                ref_images=appearance_tensor,
                                save_kv=save_kv,
                                iter_cur=-2,
                                source_masks=source_masks,
                                target_masks=target_masks,
                                pose_scale=pose_scale,
                            )[0]
                            next_original_sample = (alpha_prod_t ** 0.5) * latent_prev - (beta_prod_t ** 0.5) * model_output
                            next_original_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * latent_prev
                            next_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * next_original_epsilon
                            latents = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

                latents = latent_prev

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return MotionEditorOutput(videos=images)

    def guidance_edit(
            self,
            source_masks,
            target_masks,
            latent,
            latent_noise_ref,
            target_pose_fea,
            ref_pose_fea,
            encoder_hidden_states,
            appearance_tensor,
            t,
            up_ft_index,
            energy_scale,
            up_scale=2,
            w_fg=4.0,
            w_content=6.0,
            w_contrast=1.2,
            w_inpaint=1.2,
            up_scale_dimension=128,
    ):
        cos = nn.CosineSimilarity(dim=1)
        loss_scale = [0.5, 0.5]
        num_frames = source_masks.size()[0]

        source_masks = source_masks.unsqueeze(0)
        source_masks = rearrange(source_masks, "b f c h w -> b c f h w")
        source_masks = F.interpolate(source_masks, size=(num_frames, up_scale_dimension, up_scale_dimension), mode="nearest")

        target_masks = target_masks.unsqueeze(0)
        target_masks = rearrange(target_masks, "b f c h w -> b c f h w")
        target_masks_org = target_masks
        target_masks = F.interpolate(target_masks, size=(num_frames, up_scale_dimension, up_scale_dimension), mode="nearest")

        target_masks_org = F.interpolate(target_masks_org, size=latent.size()[-3:], mode="nearest")

        mask_over = (1-source_masks.float())*(1-target_masks.float()) > 0.5
        mask_body = source_masks.float() * (1-target_masks.float()) > 0.5
        mask_minus_gud = (1 - source_masks) > 0.5
        source_masks = source_masks > 0.5
        target_masks = target_masks > 0.5

        with torch.no_grad():
            up_ft_rec = self.estimator(
                    latent_noise_ref,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    pose_cond_fea=ref_pose_fea,
                    return_dict=False,
                    ref_images=appearance_tensor,
                    up_ft_indices=up_ft_index,
            )[0]
            # the sizes of up_ft_rec: [1, 1280, 24, 32, 32], [1, 640, 24, 64, 64]
            for f_id in range(len(up_ft_rec)):
                up_ft_rec[f_id] = F.interpolate(up_ft_rec[f_id], (num_frames, up_ft_rec[-1].shape[-2]*up_scale, up_ft_rec[-1].shape[-1]*up_scale))

        with torch.enable_grad():
            latent = latent.detach().requires_grad_(True)
            up_ft_edit = self.estimator(
                latent,
                t,
                encoder_hidden_states=encoder_hidden_states,
                pose_cond_fea=target_pose_fea,
                return_dict=False,
                ref_images=appearance_tensor,
                up_ft_indices=up_ft_index,
            )[0]
            for f_id in range(len(up_ft_rec)):
                up_ft_edit[f_id] = F.interpolate(up_ft_edit[f_id], (num_frames, up_ft_edit[-1].shape[-2] * up_scale, up_ft_edit[-1].shape[-1] * up_scale))


            loss_fg = 0
            for f_id in range(len(up_ft_rec)):
                sim_fg_list = []
                for i in range(num_frames):
                    up_ft_edit_vec = up_ft_edit[f_id][:, :, i, :, :][target_masks[:, :, i, :, :].repeat(1, up_ft_edit[f_id][:, :, i, :, :].shape[1], 1, 1)].view(up_ft_edit[f_id][:, :, i, :, :].shape[1], -1).permute(1, 0)
                    up_ft_rec_vec = up_ft_rec[f_id][:, :, i, :, :][source_masks[:, :, i, :, :].repeat(1, up_ft_rec[f_id][:, :, i, :, :].shape[1], 1, 1)].view(up_ft_rec[f_id][:, :, i, :, :].shape[1], -1).permute(1, 0)
                    up_ft_edit_vec = up_ft_edit_vec.mean(dim=0, keepdim=True)
                    up_ft_rec_vec = up_ft_rec_vec.mean(dim=0, keepdim=True)
                    sim_fg = cos(up_ft_edit_vec, up_ft_rec_vec)
                    sim_fg_list.append(sim_fg.mean())
                sim_mean = sum(sim_fg_list) / len(sim_fg_list)
                loss_fg = loss_fg + (w_fg / (1 + 4 * sim_mean)) * loss_scale[f_id]


            loss_bg = 0
            for f_id in range(len(up_ft_edit)):
                sim_body_list = []
                for i in range(num_frames):
                    sim_body = cos(up_ft_edit[f_id][:, :, i, :, :], up_ft_rec[f_id][:, :, i, :, :])[0][mask_body[0, 0, i]]
                    sim_body_list.append(sim_body.mean())
                sim_body_mean = sum(sim_body_list) / len(sim_body_list)
                loss_bg = loss_bg + w_content / (1 + 4 * sim_body_mean.mean()) * loss_scale[f_id]

            for f_id in range(len(up_ft_rec)):
                sim_overlap_list = []
                for i in range(num_frames):
                    up_ft_edit_overlap = up_ft_edit[f_id][:, :, i, :, :][mask_over[:, :, i, :, :].repeat(1, up_ft_edit[f_id][:, :, i, :, :].shape[1], 1, 1)].view(up_ft_edit[f_id][:, :, i, :, :].shape[1], -1).permute(1,0)
                    up_ft_rec_overlap = up_ft_rec[f_id][:, :, i, :, :][mask_over[:, :, i, :, :].repeat(1, up_ft_rec[f_id][:, :, i, :, :].shape[1], 1, 1)].view(up_ft_rec[f_id][:, :, i, :, :].shape[1], -1).permute(1,0)
                    sim_overlap = (cos(up_ft_edit_overlap, up_ft_rec_overlap) + 1.) / 2.
                    sim_overlap_list.append(sim_overlap.mean())
                sim_overlap_mean = sum(sim_overlap_list) / len(sim_overlap_list)
                loss_bg = loss_bg + w_contrast*sim_overlap_mean*loss_scale[f_id]

                sim_com_list = []
                for i in range(num_frames):
                    up_ft_edit_com = up_ft_edit[f_id][:, :, i, :, :][mask_body[:, :, i, :, :].repeat(1, up_ft_edit[f_id][:, :, i, :, :].shape[1],1,1)].view(up_ft_edit[f_id][:, :, i, :, :].shape[1], -1).permute(1, 0).mean(0, keepdim=True)
                    up_ft_rec_com = up_ft_rec[f_id][:, :, i, :, :][mask_minus_gud[:, :, i, :, :].repeat(1, up_ft_rec[f_id][:, :, i, :, :].shape[1],1,1)].view(up_ft_rec[f_id][:, :, i, :, :].shape[1], -1).permute(1, 0).mean(0, keepdim=True)
                    up_ft_edit_com = up_ft_edit_com.mean(dim=0, keepdim=True)
                    up_ft_rec_com = up_ft_rec_com.mean(dim=0, keepdim=True)
                    sim_com = ((cos(up_ft_edit_com, up_ft_rec_com) + 1.) / 2.)
                    sim_com_list.append(sim_com.mean())
                sim_com_mean = sum(sim_com_list) / len(sim_com_list)
                loss_bg = loss_bg + w_inpaint/(1+4*sim_com_mean)

            cond_grad_fg = torch.autograd.grad(loss_fg * energy_scale, latent, retain_graph=True)[0]
            cond_grad_bg = torch.autograd.grad(loss_bg * energy_scale, latent)[0]
            guidance = cond_grad_fg.detach() * 4e-2 * target_masks_org + cond_grad_bg.detach() * 4e-2 * (1 - target_masks_org)
            self.denoising_unet.zero_grad()
            return guidance

