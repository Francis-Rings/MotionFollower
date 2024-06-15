import os
from datetime import datetime
from pathlib import Path
import argparse
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDIMScheduler
import torch
import imageio
import numpy as np
import os.path as osp
from einops import rearrange
import torch.nn.functional as F

from src.models.attention_processor import AttnProcessor
from src.models.estimator import Estimator
from src.models.pose_guider import PoseGuider
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.motion_editor import MotionEditor
from src.pipelines.pipeline_pose2vid_collection import Pose2VideoCollectionPipeline

from src.utils.util import save_videos_grid, seed_everything, save_videos_as_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--video_root", type=str)
    parser.add_argument("--pose_root", type=str)
    parser.add_argument("--ref_pose_root", type=str)
    parser.add_argument("--source_mask_root", type=str)
    parser.add_argument("--target_mask_root", type=str)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--enable_xformers_memory_efficient_attention", type=str, default="Ture")
    parser.add_argument("--gradient_checkpointing", type=str, default="True")
    parser.add_argument("--num_inv_steps", type=int, default=50)
    parser.add_argument("--suffix", type=str, default="jpg")
    parser.add_argument("--camera", type=str, default="False")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    if args.seed is not None:
        seed_everything(args.seed)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path, ).to("cuda", dtype=weight_dtype)
    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_base_model_path,config.motion_module_path, subfolder="unet",unet_additional_kwargs=infer_config.unet_additional_kwargs, ).to(dtype=weight_dtype, device="cuda")
    estimator = Estimator.from_pretrained_2d(config.pretrained_base_model_path,
                                                  config.motion_module_path, subfolder="unet",
                                                  unet_additional_kwargs=infer_config.unet_additional_kwargs, ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(dtype=weight_dtype,
                                                                                            device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    ref_scheduler = DDIMScheduler(**sched_kwargs)

    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)

    vae.eval()
    image_enc.eval()
    denoising_unet.eval()
    pose_guider.eval()

    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H
    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    if args.gradient_checkpointing:
        denoising_unet.enable_gradient_checkpointing()

    attention_processor = AttnProcessor(camera=args.camera)
    denoising_unet.set_attn_processor(attention_processor)

    pipe = MotionEditor(
        vae=vae,
        image_encoder=image_enc,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
        estimator=estimator,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"
    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    ref_pipe = Pose2VideoCollectionPipeline(
        vae=vae,
        image_encoder=image_enc,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=ref_scheduler,
    )
    ref_pipe = ref_pipe.to("cuda", dtype=weight_dtype)

    ref_images_path = args.video_root
    pose_video_path = args.pose_root
    ref_pose_video_path = args.ref_pose_root
    source_masks_path = args.source_mask_root
    target_masks_path = args.target_mask_root

    ref_suffix = args.suffix
    ref_suffix = "." + ref_suffix
    ref_file_names = os.listdir(ref_images_path)
    # ref_image_files = [file for file in ref_file_names if file.endswith('.jpg')]
    ref_image_files = [file for file in ref_file_names if file.endswith(ref_suffix)]
    ref_image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    pose_file_names = os.listdir(pose_video_path)
    pose_files = [file for file in pose_file_names if file.endswith('.png')]
    pose_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    ref_pose_file_names = os.listdir(ref_pose_video_path)
    ref_pose_files = [file for file in ref_pose_file_names if file.endswith('.png')]
    ref_pose_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    source_mask_file_names = os.listdir(source_masks_path)
    source_mask_files = [file for file in source_mask_file_names if file.endswith('.png')]
    source_mask_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    target_mask_file_names = os.listdir(target_masks_path)
    target_mask_files = [file for file in target_mask_file_names if file.endswith('.png')]
    target_mask_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    ref_images_list = []
    ref_images_tensor_list = []
    ref_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    for ref_image_file in ref_image_files:
        ref_path = osp.join(ref_images_path, ref_image_file)
        ref_image_pil = Image.open(ref_path).convert("RGB")
        ref_images_tensor_list.append(ref_transform(ref_image_pil))
        ref_images_list.append(ref_image_pil)

    ref_images_tensor = torch.stack(ref_images_tensor_list, dim=0)  # (f, c, h, w)
    ref_images_tensor = ref_images_tensor.transpose(0, 1)

    pose_list = []
    pose_tensor_list = []
    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    for pose_file in pose_files:
        pose_path = osp.join(pose_video_path, pose_file)
        pose_image_pil = Image.open(pose_path).convert("RGB")
        pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_list.append(pose_image_pil)
    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)

    ref_pose_list = []
    ref_pose_tensor_list = []
    ref_pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    for ref_pose_file in ref_pose_files:
        ref_pose_path = osp.join(ref_pose_video_path, ref_pose_file)
        ref_pose_image_pil = Image.open(ref_pose_path).convert("RGB")
        ref_pose_tensor_list.append(ref_pose_transform(ref_pose_image_pil))
        ref_pose_list.append(ref_pose_image_pil)
    ref_pose_tensor = torch.stack(ref_pose_tensor_list, dim=0)  # (f, c, h, w)
    ref_pose_tensor = ref_pose_tensor.transpose(0, 1)

    source_mask_list = []
    target_mask_list = []
    for source_mask_file in source_mask_files:
        source_mask_path = osp.join(source_masks_path, source_mask_file)
        _source_mask = imageio.imread(source_mask_path).astype(np.float32)  ## H,W 0 and 255
        _source_mask /= 255
        source_mask_list.append(_source_mask)
    source_masks = torch.from_numpy(np.stack(source_mask_list, axis=0)).float()  # f,h,w
    source_masks = rearrange(source_masks[:, :, :, None], "f h w c -> f c h w")
    source_masks = F.interpolate(source_masks, size=(height, width), mode='nearest')
    for target_mask_file in target_mask_files:
        target_mask_path = osp.join(target_masks_path, target_mask_file)
        _target_mask = imageio.imread(target_mask_path).astype(np.float32)  ## H,W 0 and 255
        _target_mask /= 255
        target_mask_list.append(_target_mask)
    target_masks = torch.from_numpy(np.stack(target_mask_list, axis=0)).float()  # f,h,w
    target_masks = rearrange(target_masks[:, :, :, None], "f h w c -> f c h w")
    target_masks = F.interpolate(target_masks, size=(height, width), mode='nearest')

    print("Attention! We are collecting the latents from reference frames in terms of reconstruction.")
    print("You should confirm whether the initial latents of edited frames and reconstructed frames are identical")
    ref_outputs = ref_pipe(
        ref_images_list,
        ref_pose_list,
        width,
        height,
        args.L,
        args.steps,
        guidance_scale=0,
        generator=generator,
        save_kv=True,
    )
    ref_video = ref_outputs.videos
    latents_collection = ref_outputs.collections
    latents_org = ref_outputs.latents_org

    video = pipe(
        ref_images_list,
        pose_list,
        width,
        height,
        args.L,
        args.steps,
        args.cfg,
        generator=generator,
        source_masks=source_masks,
        target_masks=target_masks,
        ref_pose_images=ref_pose_list,
        ref_latents_collections=latents_collection,
        latents_org=latents_org,
        save_kv=False,
    ).videos

    edited_video = video
    pose_tensor = pose_tensor.unsqueeze(0)
    ref_images_tensor = ref_images_tensor.unsqueeze(0)
    video = torch.cat([ref_images_tensor, pose_tensor, video], dim=0)
    ref_name = os.path.basename(ref_images_path)
    pose_name = os.path.basename(pose_video_path)
    save_videos_grid(
        video,
        f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.gif",
        n_rows=3,
        fps=args.fps if args.fps is None else args.fps,
    )

    ref_pose_tensor = ref_pose_tensor.unsqueeze(0)
    ref_video = torch.cat([ref_images_tensor, ref_pose_tensor, ref_video], dim=0)
    save_videos_grid(
        ref_video,
        f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_ref.gif",
        n_rows=3,
        fps=args.fps if args.fps is None else args.fps,
    )


if __name__ == "__main__":
    main()
