"""
Anatomy-Constrained Conditional Batch Generation Script
Forces synthesis of Class 0 (Normal) chest X-rays based on custom input maps.
Saves structured outputs as: {OUTPUT_BASE_DIR}/{base_name}/{base_name}_gen_xray.png
"""

import os
import torch
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from models import SiT_models
from rela_ctrl_wrapper import SiTRelaCtrlWrapper
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args

# System architecture optimization for Ampere / Ada Lovelace
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TransportInterfaceAdapter(torch.nn.Module):
    """Enforces shape alignment between model layers and transport engine."""
    def __init__(self, execution_model, unpatchify_functional):
        super().__init__()
        self.execution_model = execution_model
        self.unpatchify_functional = unpatchify_functional

    def _enforce_structural_dimensions(self, tensor_output, reference_shape):
        if isinstance(tensor_output, tuple):
            tensor_output = tensor_output[0]
        if tensor_output.ndim == 3:
            tensor_output = self.unpatchify_functional(tensor_output)
        if tensor_output.ndim == 4 and tensor_output.size(1) == reference_shape[1] * 2:
            tensor_output, _ = tensor_output.chunk(2, dim=1)
        return tensor_output

    def forward_with_cfg(self, x, t, **kwargs):
        raw_output = self.execution_model.forward_with_cfg(x, t, **kwargs)
        return self._enforce_structural_dimensions(raw_output, x.shape)


def run_batch_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Batch Generation Engine on Device: {device}")

    latent_resolution = args.image_size // 8

    # 1. Rebuild Architecture 
    structural_backbone = SiT_models[args.model](
        input_size=latent_resolution,
        num_classes=2
    ).to(device)

    optimization_model = SiTRelaCtrlWrapper(
        base_sit_model=structural_backbone,
        condition_channels=3,
        relevant_layers=[2, 4, 6, 8, 10, 12, 14],
    ).to(device)

    # 2. Extract Checkpoint Weight States
    print(f"📦 Loading parameters from: {args.ckpt}")
    checkpoint_state = torch.load(args.ckpt, map_location=device)
    
    if "ema" in checkpoint_state:
        optimization_model.load_state_dict(checkpoint_state["ema"])
        print("✅ High-stability EMA weights loaded successfully.")
    else:
        optimization_model.load_state_dict(checkpoint_state["model"])
        print("⚠️ Warning: Using raw training weights.")
    
    optimization_model.eval()
    for param in optimization_model.parameters():
        param.requires_grad = False

    # 3. Setup Transport & High-Fidelity VAE Engine
    sanitized_interface = TransportInterfaceAdapter(optimization_model, structural_backbone.unpatchify)
    transport_infrastructure = create_transport(
        args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps
    )
    transport_sampling_engine = Sampler(transport_infrastructure)
    ode_sampler = transport_sampling_engine.sample_ode()

    vae_decoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()

    # 4. Input Map Transformation Pipeline
    mask_transform = transforms.Compose([
        transforms.Resize((latent_resolution, latent_resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # 5. Discover maps using the custom string-parsing pattern
    all_files = os.listdir(args.map_dir)
    maps = [f for f in all_files if "_organ_mask" in f.lower()]
    print(f"🔎 Discovered {len(maps)} target anatomical maps inside MAP_DIR.")

    # Freeze seed for consistent metric evaluation safety
    torch.manual_seed(args.inference_seed)

    # 6. Generation Loop
    for idx, map_filename in enumerate(tqdm(maps, desc="Generating Normal X-Rays")):
        map_path = os.path.join(args.map_dir, map_filename)
        
        # Clean up filename to extract unique ID using user pattern
        try:
            print(f"Using mask{map_path}")
            study_id = map_filename.split("_")[0]
            category = map_filename.split(study_id + "_")[1].split("organ_mask.png")[0]
            category = category[:-1]
        except ValueError:
            print(f"⚠️ Could not parse base name from file: {map_filename}. Skipping.")
            continue


        output_file_path = os.path.join(args.output_base_dir, f"{study_id}_SiT2_{category}.png")
        print(f"Saving file for..{output_file_path}")
        # Skip execution if item already exists (Supports safe interruption/resumption)
        if os.path.exists(output_file_path):
            continue


        # Process and prepare the map tensor
        map_pil = Image.open(map_path).convert('RGB')
        mask_tensor_latent = mask_transform(map_pil).unsqueeze(0).to(device)

        # Prepare CFG batch split profiles forcing Class 0 (Normal)
        noise_seed = torch.randn(1, 4, latent_resolution, latent_resolution, device=device)
        cfg_noise = torch.cat([noise_seed, noise_seed], dim=0)
        cfg_masks = torch.cat([mask_tensor_latent, mask_tensor_latent], dim=0)
        cfg_labels = torch.tensor([0, 0], dtype=torch.long, device=device) # [0: No_Finding]

        sampling_parameters = dict(
            y=cfg_labels,
            condition_img=cfg_masks,
            cfg_scale=args.cfg_scale
        )

        with torch.no_grad():
            # Sample through the ODE path solver
            generated_latents = ode_sampler(cfg_noise, sanitized_interface.forward_with_cfg, **sampling_parameters)[-1]
            
            # Isolate the guided batch from unguided drop components
            generated_latents, _ = generated_latents.chunk(2, dim=0)
            
            # Decode out of latent tensor coordinates via VAE 
            decoded_image = vae_decoder.decode(generated_latents / 0.18215).sample
            decoded_image = (decoded_image.clamp(-1, 1) + 1) / 2.0

        # Save single crisp full-resolution image directly into nested destination path
        save_image(decoded_image[0], output_file_path, normalize=False)

    print(f"\n✨ Batch compilation completed. Files outputted safely to: {args.output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Normal Generation.")
    parser.add_argument("--map-dir",         type=str, default=r"C:\Users\HCI-4\Desktop\XRay_Generation\Metrics_evaluation\25_09_FID_MSSIM\masks")
    parser.add_argument("--output-base-dir", type=str, default=r"C:\Users\HCI-4\Desktop\XRay_Generation\Metrics_evaluation\25_09_FID_MSSIM\SiT_2")
    parser.add_argument("--ckpt",            type=str, default=r"C:\Users\HCI-4\Desktop\SiTXRay\SiT\results_relactrl_production\001-SiTRelaCtrl-Linear-velocity\checkpoints\best_structural_model.pt", help="Path to your best_structural_model.pt")
    parser.add_argument("--model",           type=str, default="SiT-XL/2")
    parser.add_argument("--image-size",      type=int, default=256)
    parser.add_argument("--vae",             type=str, default="mse")
    parser.add_argument("--cfg-scale",       type=float, default=4.0)
    parser.add_argument("--inference-seed",  type=int, default=42)
    
    parse_transport_args(parser)
    run_batch_inference(parser.parse_args())


