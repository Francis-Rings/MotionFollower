from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
from . import ptp_utils
from . import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from einops import rearrange
from transformers import CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor

LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
MAX_NUM_WORDS = 77
device = torch.device('cuda')
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_path = "/checkpoints/stable-diffusion-v1-5"

ldm_stable = None
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")


class MyNullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context, normal_infer=False):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context, normal_infer=False)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None, normal_infer=False):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else 7.5
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context, normal_infer=False)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # (1, 77, 768)
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        # (2, 77, 768)
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def init_signals(self, ref_images_list, ref_pose_list):
        clip_images_list = []
        for ref_clip_image in ref_images_list:
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

        ref_images_list = []
        for ref_image in ref_images_list:
            ref_image_tensor = self.ref_image_processor.preprocess(
                ref_image, height=512, width=512
            )  # (bs, c, width, height)
            ref_images_list.append(ref_image_tensor)
        ref_images_tensor = torch.stack(ref_images_list, dim=0)
        ref_images_tensor = ref_images_tensor.squeeze()
        ref_images_tensor = ref_images_tensor.unsqueeze(0)
        appearance_tensor = ref_images_tensor.to(dtype=self.model.dtype, device=self.model.device)

        target_pose_cond_tensor_list = []
        for target_pose_image in ref_pose_list:
            target_pose_cond_tensor = self.cond_image_processor.preprocess(target_pose_image, height=height,
                                                                           width=width)
            target_pose_cond_tensor = target_pose_cond_tensor.squeeze().unsqueeze(1)
            target_pose_cond_tensor_list.append(target_pose_cond_tensor)

        target_pose_cond_tensor = torch.cat(target_pose_cond_tensor_list, dim=1)  # (c, t, h, w)
        target_pose_cond_tensor = target_pose_cond_tensor.unsqueeze(0)
        target_pose_cond_tensor = target_pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        target_pose_fea = self.pose_guider(target_pose_cond_tensor)

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        # cond = cond_embeddings if self.null_inv_with_prompt else uncond_embeddings
        cond = cond_embeddings
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            # noise_pred = self.get_noise_pred_single(latent, t, cond, normal_infer=True)
            noise_pred = self.get_noise_pred_single(latent, t, cond, normal_infer=False)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, latent):
        ddim_latents = self.ddim_loop(latent)
        return ddim_latents

    def null_optimization(self, latents, null_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=null_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, normal_infer=False)
            for j in range(null_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings, normal_infer=False)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                assert not torch.isnan(uncond_embeddings.abs().mean())
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, null_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, normal_infer=False)
        bar.close()
        return uncond_embeddings_list

    def invert(self, latents: torch.Tensor, ref_images_list=None, ref_pose_list=None, null_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, null_base_lr=1e-2):
        self.init_signals(ref_images_list, ref_pose_list)
        if verbose:
            print("DDIM inversion...")
        ddim_latents = self.ddim_inversion(latents.to(torch.float32))
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, null_inner_steps, early_stop_epsilon)
        return ddim_latents[-1], uncond_embeddings

    def __init__(self, model, guidance_scale, null_inv_with_prompt, null_normal_infer=False, image_encoder=None):
        self.null_normal_infer = null_normal_infer
        self.null_inv_with_prompt = null_inv_with_prompt
        self.guidance_scale = guidance_scale
        self.model = model
        self.image_encoder = image_encoder
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=8, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_convert_rgb=True,
            do_normalize=False,
        )
