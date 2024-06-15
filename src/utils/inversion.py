from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np


class DDIMInversion:
    def __init__(self, model, scheduler, NUM_DDIM_STEPS):
        self.model = model
        self.scheduler = scheduler
        self.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS

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

    def get_noise_pred_single(self, latents, t, context, ref_images_pil=None, pose_cond_fea=None):
        noise_pred = self.model(
            latents,
            t,
            pose_cond_fea=pose_cond_fea,
            encoder_hidden_states=context,
            ref_images=ref_images_pil)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_emb_img(self, clip_emb_im=None):
        self.emb_im = clip_emb_im.to(self.model.device)

    @torch.no_grad()
    def ddim_loop(self, latent, ref_images_pil=None, pose_cond_fea=None):
        cond_embeddings = self.emb_im
        all_latent = [latent]
        latent = latent.clone().detach()
        print('DDIM Inversion:')
        for i in tqdm(range(self.NUM_DDIM_STEPS)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, ref_images_pil=ref_images_pil, pose_cond_fea=pose_cond_fea)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)

        return all_latent

    def invert(self, ddim_latents, clip_emb_im=None, ref_images_pil=None, pose_cond_fea=None):
        self.init_emb_img(clip_emb_im=clip_emb_im)
        ddim_latents = self.ddim_loop(ddim_latents, ref_images_pil=ref_images_pil, pose_cond_fea=pose_cond_fea)
        return ddim_latents