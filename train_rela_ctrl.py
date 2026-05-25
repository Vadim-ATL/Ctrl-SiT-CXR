import os
import platform
import argparse
import logging
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from models import sit_models
from rela_ctrl_wrapper import SitRelaCtrlWrapper
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)

def _win_get_world_size(): return 1
def _win_get_rank(): return 0
def _win_barrier(): return None
def _win_all_reduce(tensor, op=None): return None
def _win_all_gather_into_tensor(output_tensor, input_tensor): output_tensor.copy_(input_tensor)
def _win_destroy_process_group(): return None
def _win_is_initialized(): return True

def cache_latents_if_needed(data_path, image_size, vae_name, device):
    images_dir = os.path.join(data_path, "images")
    latents_dir = os.path.join(data_path, "latents")
    
    existing_latents = glob(os.path.join(latents_dir, "*", "*.pt"))
    if len(existing_latents) > 0:
        return latents_dir

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device)
    vae.eval()
    
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    classes = ["No_Finding", "Pneumonia"]
    for class_name in classes:
        src_folder = os.path.join(images_dir, class_name)
        tgt_folder = os.path.join(latents_dir, class_name)
        os.makedirs(tgt_folder, exist_ok=True)
        
        if not os.path.exists(src_folder):
            continue
            
        images = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        with torch.no_grad():
            for img_name in tqdm(images):
                img_path = os.path.join(src_folder, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                latent = vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)
                latent = latent.squeeze(0).cpu()
                
                save_name = os.path.splitext(img_name)[0] + ".pt"
                torch.save(latent, os.path.join(tgt_folder, save_name))
                
    del vae
    torch.cuda.empty_cache()
    return latents_dir

class LatentXRayMaskDataset(Dataset):
    def __init__(self, latents_dir, masks_dir, image_size):
        self.samples = []
        self.classes = ["No_Finding", "Pneumonia"]
        
        for class_idx, class_name in enumerate(self.classes):
            latent_class_dir = os.path.join(latents_dir, class_name)
            mask_class_dir = os.path.join(masks_dir, class_name)
            
            if not os.path.exists(latent_class_dir) or not os.path.exists(mask_class_dir):
                continue
                
            for file_name in os.listdir(latent_class_dir):
                if file_name.endswith('.pt'):
                    base_name = os.path.splitext(file_name)[0]
                    mask_path = None
                    for ext in ['.png', '.jpg', '.jpeg']:
                        temp_path = os.path.join(mask_class_dir, base_name + "_organ_mask" + ext)
                        if os.path.exists(temp_path):
                            mask_path = temp_path
                            break
                    
                    if mask_path:
                        latent_path = os.path.join(latent_class_dir, file_name)
                        self.samples.append((latent_path, mask_path, class_idx))
                        
        latent_size = image_size // 8
        self.mask_transform = transforms.Compose([
            transforms.Resize((latent_size, latent_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() 
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        latent_path, mask_path, label = self.samples[idx]
        latent = torch.load(latent_path)
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor = self.mask_transform(mask_img)
        return latent, mask_tensor, label

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main(args):
    if platform.system() == "Windows":
        dist.get_world_size = _win_get_world_size
        dist.get_rank = _win_get_rank
        dist.barrier = _win_barrier
        dist.all_reduce = _win_all_reduce
        dist.all_gather_into_tensor = _win_all_gather_into_tensor
        dist.destroy_process_group = _win_destroy_process_group
        dist.is_initialized = _win_is_initialized
    else:
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    if rank == 0:
        latents_dir = cache_latents_if_needed(args.data_path, args.image_size, args.vae, device)
    dist.barrier()
    
    if rank != 0:
        latents_dir = os.path.join(args.data_path, "latents")

    samples_dir = ""
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-{model_string_name}-{args.path_type}-{args.prediction}-{args.loss_weight}-Latent"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        samples_dir = f"{experiment_dir}/samples"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir)
        if args.wandb:
            wandb_utils.initialize(args, os.environ.get("entity", "default_entity"), experiment_name, os.environ.get("project", "default_project"))
    else:
        logger = create_logger(None)

    latent_size = args.image_size // 8
    base_model = sit_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    
    model_weights = None
    ema_weights = None

    if args.ckpt is not None:
        state_dict = find_model(args.ckpt)
        if "model" in state_dict:
            model_weights = state_dict["model"]
            ema_weights = state_dict["ema"]
        else:
            model_weights = state_dict
            ema_weights = state_dict
            
        model_dict = base_model.state_dict()
        filtered_model = {k: v for k, v in model_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_model)
        base_model.load_state_dict(model_dict)

    model = SitRelaCtrlWrapper(
        base_sit_model=base_model,
        condition_channels=1, 
        relevant_layers=[2, 4, 6, 8, 10, 12, 14]
    ).to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    if args.ckpt is not None and ema_weights is not None:
        base_ema_dict = ema.base_model.state_dict()
        filtered_ema = {k: v for k, v in ema_weights.items() if k in base_ema_dict and v.shape == base_ema_dict[k].shape}
        base_ema_dict.update(filtered_ema)
        ema.base_model.load_state_dict(base_ema_dict)

    if platform.system() == "Windows":
        class DummyDistributedDataParallel(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)
        model = DummyDistributedDataParallel(model.to(device))
    else:
        model = DistributedDataParallel(model.to(device), device_ids=[device])

    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    transport_sampler = Sampler(transport)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    requires_grad(vae, False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0)

    masks_dir = os.path.join(args.data_path, "masks")
    dataset = LatentXRayMaskDataset(latents_dir=latents_dir, masks_dir=masks_dir, image_size=args.image_size)

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=local_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    use_cfg = args.cfg_scale > 1.0
    n = local_batch_size
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    
    val_masks = []
    for i in range(n):
        _, mask_tensor, _ = dataset[i]
        val_masks.append(mask_tensor)

    fixed_condition_masks = torch.stack(val_masks).to(device)

    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        cond_input = torch.cat([fixed_condition_masks, fixed_condition_masks], 0)
        ys = torch.zeros(n * 2, dtype=torch.long, device=device) 
        sample_model_kwargs = dict(y=ys, condition_img=cond_input, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        ys = torch.zeros(n, dtype=torch.long, device=device)
        sample_model_kwargs = dict(y=ys, condition_img=fixed_condition_masks)
        model_fn = ema.forward

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for x, condition_img, y in loader:
            x = x.to(device)
            condition_img = condition_img.to(device)
            y = y.to(device)
            
            model_kwargs = dict(y=y, condition_img=condition_img)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if args.wandb:
                    wandb_utils.log({ "train loss": avg_loss, "train steps/sec": steps_per_sec }, step=train_steps)
                running_loss = 0
                log_steps = 0
                start_time = time()

        current_epoch_count = epoch + 1
        if current_epoch_count % 20 == 0:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/latest_checkpoint.pt"
                torch.save(checkpoint, checkpoint_path)
            dist.barrier()
            
            if rank == 0:
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    
                    samples = vae.decode(samples / 0.18215).sample
                    out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                    if rank == 0:
                        sample_path = f"{samples_dir}/epoch_{current_epoch_count:05d}.png"
                        save_image(out_samples, sample_path, nrow=8, normalize=True, value_range=(-1, 1))
            dist.barrier()

    model.eval()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(sit_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)