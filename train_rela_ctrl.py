"""
Anatomy-Constrained Conditional Latent Diffusion Training Framework
Optimized for Structural Regulation via Relation Control (ReLaCtrl) Blocks.
Targets dual-class pathology optimization (0: No_Finding, 1: Pneumonia).
Enhanced with Batch-Level Structural Weighting and Dynamic Plateau Adaptivity.
"""

import os
import csv
import logging
import platform
from time import time
from glob import glob
from copy import deepcopy

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models import SiT_models                        
from rela_ctrl_wrapper import SiTRelaCtrlWrapper     
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args

# Optimize CUDA operations for modern Ampere/Ada Lovelace architectures
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


class TransportInterfaceAdapter(torch.nn.Module):
    """
    Interface adapter designed to enforce strict dimensional compliance 
    between downstream custom wrappers and external transport layer logic.
    """
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

    def forward(self, x, t, **kwargs):
        raw_output = self.execution_model(x, t, **kwargs)
        return self._enforce_structural_dimensions(raw_output, x.shape)

    def forward_with_cfg(self, x, t, **kwargs):
        raw_output = self.execution_model.forward_with_cfg(x, t, **kwargs)
        return self._enforce_structural_dimensions(raw_output, x.shape)


class IdentityDistributedParallel(torch.nn.Module):
    """
    Fallback context wrapper mapping execution models uniformly within 
    single-device environments while maintaining interface parity with DDP.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# Distributed Execution Environment Stubs for Windows Compatibility
def _runtime_get_world_size(): return 1
def _runtime_get_rank(): return 0
def _runtime_barrier(): return None


class LatentAnatomyDataset(Dataset):
    """
    Optimized data engine handling aligned execution elements:
    Cached Tensor Latents, Structural Control Maps, and Categorical Ground Truths.
    """
    def __init__(self, manifest_samples, latents_directory, target_resolution):
        self.samples = manifest_samples
        self.latents_directory = latents_directory
        spatial_dimension = target_resolution // 8
        self.mask_operator = transforms.Compose([
            transforms.Resize((spatial_dimension, spatial_dimension),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target_sample = self.samples[idx]
        latent_tensor_path = os.path.join(self.latents_directory, f"{target_sample['dicom_id']}.pt")
        latent_tensor = torch.load(latent_tensor_path, weights_only=True)
        
        mask_image = Image.open(target_sample['mask_path']).convert('RGB')
        mask_tensor = self.mask_operator(mask_image)
        
        return latent_tensor, mask_tensor, target_sample['label']


def parse_and_validate_manifest(base_data_path):
    manifest_csv = os.path.join(base_data_path, "master_multipathology_manifest.csv")
    source_images = os.path.join(base_data_path, "local_images")
    structural_maps = os.path.join(base_data_path, "local_maps", "local_maps")
    
    verified_records = []
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Target system manifest missing at path: {manifest_csv}")

    with open(manifest_csv, mode='r', encoding='utf-8') as stream:
        dictionary_reader = csv.DictReader(stream)
        for line_item in dictionary_reader:
            dicom_identifier = line_item.get('dicom_id')
            if not dicom_identifier:
                continue
            
            try:
                null_finding_flag = float(line_item.get('No Finding') or 0)
                pathology_pneumonia_flag = float(line_item.get('Pneumonia') or 0)
            except ValueError:
                continue

            if int(null_finding_flag) == 1:
                assigned_class = 0
            elif int(pathology_pneumonia_flag) == 1:
                assigned_class = 1
            else:
                continue

            image_file_path = os.path.join(source_images, f"{dicom_identifier}.jpg")
            mask_file_path = os.path.join(structural_maps, f"{dicom_identifier}_anatomic_map.png")

            if os.path.exists(image_file_path) and os.path.exists(mask_file_path):
                verified_records.append({
                    'dicom_id': dicom_identifier,
                    'img_path': image_file_path,
                    'mask_path': mask_file_path,
                    'label': assigned_class
                })
                
    return verified_records


def execute_latent_caching_pass(records, data_path, resolution, vae_identifier, execution_device):
    cache_directory = os.path.join(data_path, "local_latents")
    os.makedirs(cache_directory, exist_ok=True)
    
    discovered_caches = glob(os.path.join(cache_directory, "*.pt"))
    if len(discovered_caches) >= len(records) and len(records) > 0:
        return cache_directory

    vae_model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_identifier}").to(execution_device).eval()
    normalization_pipeline = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    with torch.no_grad():
        for record in tqdm(records, desc="Executing Latent Cache Generation"):
            target_path = os.path.join(cache_directory, f"{record['dicom_id']}.pt")
            if os.path.exists(target_path):
                continue
            raw_pixel_tensor = normalization_pipeline(Image.open(record['img_path']).convert('RGB')).unsqueeze(0).to(execution_device)
            latent_space_distribution = vae_model.encode(raw_pixel_tensor).latent_dist.sample()
            scaled_latent_tensor = latent_space_distribution.mul_(0.18215).squeeze(0).cpu()
            torch.save(scaled_latent_tensor, target_path)
            
    del vae_model
    torch.cuda.empty_cache()
    return cache_directory


@torch.no_grad()
def SynchronizeExponentialMovingAverage(target_ema_model, training_source_model, decay_factor=0.9999):
    for (source_name, source_param), (_, ema_param) in zip(training_source_model.named_parameters(), target_ema_model.named_parameters()):
        ema_param.mul_(decay_factor).add_(source_param.data, alpha=1 - decay_factor)


def configure_parameter_gradients(model_hierarchy, enable_gradients=True):
    for parameter in model_hierarchy.parameters():
        parameter.requires_grad = enable_gradients


def instantiate_system_logger(output_directory):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_directory}/production_execution.log")],
    )
    return logging.getLogger(__name__)


def main(hyperparameters):
    assert torch.cuda.is_available(), "Critical Error: Execution requires operational CUDA environment."

    # Initialize Distributed Run Environments
    if platform.system() == "Windows":
        import torch.distributed as active_distribution
        active_distribution.get_world_size = _runtime_get_world_size
        active_distribution.get_rank = _runtime_get_rank
        active_distribution.barrier = _runtime_barrier
    else:
        import torch.distributed as active_distribution
        active_distribution.init_process_group("nccl")

    system_rank = active_distribution.get_rank()
    target_device_id = system_rank % torch.cuda.device_count()
    computed_local_batch_size = hyperparameters.global_batch_size // active_distribution.get_world_size()
    
    torch.manual_seed(hyperparameters.global_seed)
    torch.cuda.set_device(target_device_id)

    # Output directory scaffolding
    if system_rank == 0:
        os.makedirs(hyperparameters.results_dir, exist_ok=True)
        allocation_index = len(glob(f"{hyperparameters.results_dir}/*"))
        experiment_identifier = f"{allocation_index:03d}-SiTRelaCtrl-{hyperparameters.path_type}-{hyperparameters.prediction}"
        experiment_path = f"{hyperparameters.results_dir}/{experiment_identifier}"
        checkpoint_path = f"{experiment_path}/checkpoints"
        visualization_path = f"{experiment_path}/diagnostics"
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(visualization_path, exist_ok=True)
        system_logger = instantiate_system_logger(experiment_path)
        system_logger.info(f"Initialized Experiment Path: {experiment_path}")
    else:
        system_logger = logging.getLogger(__name__)
        system_logger.addHandler(logging.NullHandler())

    curated_records = parse_and_validate_manifest(hyperparameters.data_path)
    if not curated_records:
        raise ValueError("Data Validation Failure: Zero matching pairs extracted from manifest.")

    if system_rank == 0:
        latents_directory_path = execute_latent_caching_pass(
            curated_records, hyperparameters.data_path, hyperparameters.image_size, hyperparameters.vae, target_device_id
        )
    active_distribution.barrier()
    if system_rank != 0:
        latents_directory_path = os.path.join(hyperparameters.data_path, "local_latents")

    inferred_latent_resolution = hyperparameters.image_size // 8

    # Base Architecture Deployment
    structural_backbone = SiT_models[hyperparameters.model](
        input_size=inferred_latent_resolution,
        num_classes=2                                    
    ).to(target_device_id)

    if hyperparameters.ckpt and not hyperparameters.resume:
        resolved_checkpoint = find_model(hyperparameters.ckpt)
        extracted_weights = resolved_checkpoint.get("ema", resolved_checkpoint.get("model", resolved_checkpoint))
        backbone_state_map = structural_backbone.state_dict()
        compliant_weights = {k: v for k, v in extracted_weights.items()
                             if k in backbone_state_map and v.shape == backbone_state_map[k].shape}
        backbone_state_map.update(compliant_weights)
        structural_backbone.load_state_dict(backbone_state_map)
        system_logger.info(f"Loaded Core Backbone Model: {len(compliant_weights)} parameters synchronized.")

    # Relation Control Structural Wrapper Integration
    optimization_model = SiTRelaCtrlWrapper(                    
        base_sit_model=structural_backbone,
        condition_channels=3,
        relevant_layers=[2, 4, 6, 8, 10, 12, 14],
    ).to(target_device_id)

    active_gradients = [p for p in optimization_model.parameters() if p.requires_grad]
    system_logger.info(f"Active Trainable Parameters: {sum(p.numel() for p in active_gradients):,} | "
                       f"Frozen Base Weight Matrices: {sum(p.numel() for p in structural_backbone.parameters()):,}")

    historical_ema_model = deepcopy(optimization_model).to(target_device_id)
    configure_parameter_gradients(historical_ema_model, False)
    SynchronizeExponentialMovingAverage(historical_ema_model, optimization_model, decay_factor=0)

    if platform.system() == "Windows":
        execution_ddp_wrapper = IdentityDistributedParallel(optimization_model)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        execution_ddp_wrapper = DDP(optimization_model, device_ids=[target_device_id])

    # Transport Layer Configurations
    sanitized_training_interface = TransportInterfaceAdapter(execution_ddp_wrapper, structural_backbone.unpatchify)
    sanitized_evaluation_interface = TransportInterfaceAdapter(historical_ema_model, structural_backbone.unpatchify)

    transport_infrastructure = create_transport(
        hyperparameters.path_type, hyperparameters.prediction, hyperparameters.loss_weight,
        hyperparameters.train_eps, hyperparameters.sample_eps
    )
    transport_sampling_engine = Sampler(transport_infrastructure)

    # Diagnostic VAE Engine Configuration
    vae_decoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{hyperparameters.vae}").to(target_device_id).eval()
    configure_parameter_gradients(vae_decoder, False)

    system_optimizer = torch.optim.AdamW(active_gradients, lr=hyperparameters.lr, weight_decay=0)
    
    # ── PROGRESSIVE PLATEAU SCHEDULER CONFIGURATION ──
    execution_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        system_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # ── CHECKPOINT RESUMPTION LIFECYCLE ENGINE ──
    starting_epoch_idx = 0
    if hyperparameters.resume:
        if os.path.isfile(hyperparameters.resume):
            system_logger.info(f"Targeting recovery checkpoint: '{hyperparameters.resume}'")
            checkpoint_state = torch.load(hyperparameters.resume, map_location=f"cuda:{target_device_id}")
            
            # Map structural weights uniform across multi-device execution wrappers
            optimization_model.load_state_dict(checkpoint_state["model"])
            historical_ema_model.load_state_dict(checkpoint_state["ema"])
            system_optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            
            starting_epoch_idx = checkpoint_state["epoch"] + 1
            system_logger.info(f"Recovery mapping verified. Re-entering execution stream at Epoch: {starting_epoch_idx}")
            del checkpoint_state
            torch.cuda.empty_cache()
        else:
            raise FileNotFoundError(f"Recovery target file path signature missing: '{hyperparameters.resume}'")

    target_dataset = LatentAnatomyDataset(curated_records, latents_directory_path, hyperparameters.image_size)
    execution_sampler = DistributedSampler(
        target_dataset, num_replicas=active_distribution.get_world_size(),
        rank=system_rank, shuffle=True, seed=hyperparameters.global_seed
    )
    data_loader = DataLoader(
        target_dataset, batch_size=computed_local_batch_size, shuffle=False,
        sampler=execution_sampler, num_workers=hyperparameters.num_workers,
        pin_memory=True, drop_last=True
    )
    system_logger.info(f"Dataset Verified. Volume: {len(target_dataset)} samples | Total Batches per Epoch: {len(data_loader)}")

    # Isolate Constant Diagnostic Batch For Multi-Column Analysis Tracking
    diagnostic_latents, diagnostic_masks, diagnostic_labels = [], [], []
    for step_idx in range(min(computed_local_batch_size, len(target_dataset))):
        lat_instance, mask_instance, label_instance = target_dataset[step_idx]
        diagnostic_latents.append(lat_instance)
        diagnostic_masks.append(mask_instance)
        diagnostic_labels.append(label_instance)
    diagnostic_latents = torch.stack(diagnostic_latents).to(target_device_id)
    diagnostic_masks = torch.stack(diagnostic_masks).to(target_device_id)
    diagnostic_labels = torch.tensor(diagnostic_labels, dtype=torch.long, device=target_device_id)

    noise_latents_seed = torch.randn_like(diagnostic_latents)
    execute_cfg = hyperparameters.cfg_scale > 1.0
    
    if execute_cfg:
        stratified_noise = torch.cat([noise_latents_seed, noise_latents_seed], 0)
        stratified_masks = torch.cat([diagnostic_masks, diagnostic_masks], 0)
        stratified_labels = torch.cat([diagnostic_labels, diagnostic_labels], 0)
        sampling_parameters = dict(y=stratified_labels, condition_img=stratified_masks, cfg_scale=hyperparameters.cfg_scale)
        functional_sampling_target = sanitized_evaluation_interface.forward_with_cfg
    else:
        stratified_noise = noise_latents_seed
        sampling_parameters = dict(y=diagnostic_labels, condition_img=diagnostic_masks)
        functional_sampling_target = sanitized_evaluation_interface.forward

    optimal_monitored_loss = float("inf")
    accumulated_training_steps = starting_epoch_idx * len(data_loader)
    period_running_loss = 0.0
    logged_steps_count = 0
    temporal_anchor = time()

    system_logger.info(f"Initiating Training Paradigm Across {hyperparameters.epochs} Epochs...")

    for epoch_idx in range(starting_epoch_idx, hyperparameters.epochs):
        execution_ddp_wrapper.train()
        execution_sampler.set_epoch(epoch_idx)

        for batch_x, batch_condition_mask, batch_y in data_loader:
            batch_x = batch_x.to(target_device_id)
            batch_condition_mask = batch_condition_mask.to(target_device_id)
            batch_y = batch_y.to(target_device_id)

            forward_arguments = dict(y=batch_y, condition_img=batch_condition_mask)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_state = transport_infrastructure.training_losses(sanitized_training_interface, batch_x, forward_arguments)
                base_loss_tensor = loss_state["loss"]  # Returns 1D Tensor: [Batch_Size]
                
                # ── BATCH-LEVEL ANATOMICAL STRUCTURAL WEIGHTING ENGINE ──
                # Compute mask density profile per sample across spatial layers [B]
                mask_density = batch_condition_mask.view(batch_condition_mask.size(0), -1).mean(dim=1)
                
                # Calculate penalty multiplier based on region importance
                spatial_loss_modifier = 1.0 + (hyperparameters.anatomy_loss_multiplier * mask_density)
                
                # Apply structural constraint mapping directly to the reduced loss vector
                minimized_loss = (base_loss_tensor * spatial_loss_modifier).mean()

            system_optimizer.zero_grad()
            minimized_loss.backward()
            
            # Active stabilization via gradient clipping
            torch.nn.utils.clip_grad_norm_(active_gradients, 1.0)
            system_optimizer.step()
            
            SynchronizeExponentialMovingAverage(historical_ema_model, execution_ddp_wrapper.module)

            period_running_loss += minimized_loss.item()
            logged_steps_count += 1
            accumulated_training_steps += 1

            if accumulated_training_steps % hyperparameters.log_every == 0:
                mean_loss_value = period_running_loss / logged_steps_count
                processing_velocity = logged_steps_count / (time() - temporal_anchor)
                current_lr = system_optimizer.param_groups[0]['lr']
                system_logger.info(f"[Step {accumulated_training_steps:06d} | Epoch {epoch_idx:03d}] "
                                   f"Loss State: {mean_loss_value:.5f} | LR: {current_lr:.2e} | Speed: {processing_velocity:.1f} steps/sec")
                period_running_loss = 0.0
                logged_steps_count = 0
                temporal_anchor = time()

        computed_epoch_loss = period_running_loss / max(logged_steps_count, 1)
        
        # Step dynamic scheduler with current average loss profile
        execution_scheduler.step(computed_epoch_loss)

        if system_rank == 0 and computed_epoch_loss < optimal_monitored_loss and computed_epoch_loss > 0:
            optimal_monitored_loss = computed_epoch_loss
            torch.save({
                "epoch": epoch_idx, 
                "model": execution_ddp_wrapper.module.state_dict(),
                "ema": historical_ema_model.state_dict(), 
                "optimizer_state_dict": system_optimizer.state_dict(),
                "train_loss": optimal_monitored_loss,
            }, f"{checkpoint_path}/best_structural_model.pt")

        # ── THREE-COLUMN DIAGNOSTIC VISUALIZATION PIPELINE ──
        if system_rank == 0 and (epoch_idx + 1) % hyperparameters.visualization_interval == 0:
            execution_ddp_wrapper.eval()
            with torch.no_grad():
                ode_sampler_function = transport_sampling_engine.sample_ode()
                generated_output_latents = ode_sampler_function(stratified_noise, functional_sampling_target, **sampling_parameters)[-1]
                if execute_cfg:
                    generated_output_latents, _ = generated_output_latents.chunk(2, dim=0)
                
                # Column 1: Reconstruction of Original Source Signal via Latent Space Decoding
                original_images_decoded = vae_decoder.decode(diagnostic_latents / 0.18215).sample
                original_images_decoded = (original_images_decoded.clamp(-1, 1) + 1) / 2
                
                # Column 2: Upsampled Conditioning Structural Anatomy Map
                structural_masks_normalized = torch.nn.functional.interpolate(
                    diagnostic_masks, size=(hyperparameters.image_size, hyperparameters.image_size), mode='nearest'
                )
                
                # Column 3: Generated Inference Output Matrix
                inferred_synthesis_decoded = vae_decoder.decode(generated_output_latents / 0.18215).sample
                inferred_synthesis_decoded = (inferred_synthesis_decoded.clamp(-1, 1) + 1) / 2
                
                # Horizontal Matrix Synthesis Assembly
                synchronized_diagnostic_rows = []
                for sample_idx in range(diagnostic_latents.size(0)):
                    evaluation_triplet = torch.cat([
                        original_images_decoded[sample_idx], 
                        structural_masks_normalized[sample_idx], 
                        inferred_synthesis_decoded[sample_idx]
                    ], dim=2)
                    synchronized_diagnostic_rows.append(evaluation_triplet)
                
                consolidated_evaluation_sheet = torch.stack(synchronized_diagnostic_rows, dim=0)
                save_image(
                    consolidated_evaluation_sheet, 
                    f"{visualization_path}/epoch_{epoch_idx+1:04d}_validation_comparison.png",
                    nrow=1, normalize=False
                )
                system_logger.info(f"Diagnostic Analysis Sheet Extracted for Epoch {epoch_idx+1}.")
            execution_ddp_wrapper.train()

    system_logger.info("Training Run Concluded Successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Production Flow Matching Training Framework with ReLaCtrl Tuning.")
    parser.add_argument("--data-path",                  type=str,   required=True, help="Path to raw assets root directory.")
    parser.add_argument("--results-dir",                type=str,   default="results_relactrl_production")
    parser.add_argument("--model",                      type=str,   default="SiT-XL/2", choices=list(SiT_models.keys()))
    parser.add_argument("--image-size",                 type=int,   default=256, choices=[256, 512])
    parser.add_argument("--epochs",                     type=int,   default=100)
    parser.add_argument("--global-batch-size",          type=int,   default=16)
    parser.add_argument("--global-seed",                type=int,   default=0)
    parser.add_argument("--vae",                        type=str,   default="mse", choices=["ema", "mse"])
    parser.add_argument("--num-workers",                type=int,   default=4)
    parser.add_argument("--log-every",                  type=int,   default=50)
    parser.add_argument("--visualization-interval",     type=int,   default=5, help="Epoch sequence step to trigger grid compilation.")
    parser.add_argument("--anatomy-loss-multiplier",    type=float, default=2.5, help="Multiplier factor penalizing deviation inside mask anatomy boundaries.")
    parser.add_argument("--dice-loss-weight",           type=float, default=0.5, help="Unused parameter retained for legacy CLI argument backward-compatibility.")
    parser.add_argument("--cfg-scale",                  type=float, default=4.0)
    parser.add_argument("--lr",                         type=float, default=1e-4)
    parser.add_argument("--ckpt",                       type=str,   default=None, help="Path to pre-trained foundation weights.")
    parser.add_argument("--resume",                     type=str,   default=None, help="Path to mid-run checkpoint .pt file for model state recovery.")

    parse_transport_args(parser)
    main(parser.parse_args())