"""Main training runner for AudioAuth.

Implements iteration-based training with GAN losses, distributed training support,
and comprehensive logging.
"""

import os
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .config import Config
from .models import Generator, Detector
from .models.watermarking import AudioWatermarking
from .losses import (
    MelSpectrogramLoss,
    MultiScaleSTFTLoss,
    GANLoss,
    LocalizationLoss,
    DecodingLoss,
    L1Loss,
    TFLoudnessLoss
)
from .dataset import AudioDataset, collater
from .utils import IterLoader
from .optims import (
    get_optimizers_and_schedulers,
    apply_gradient_clipping,
    OptimizerConfig,
    SchedulerConfig
)
from .exceptions import ValidationError, ConfigError
from .logger import setup_logger, MetricLogger
from .metrics import (
    compute_audio_reconstruction_metrics,
    compute_localization_metrics,
    compute_detection_metrics
)
from .checkpoint_utils import (
    save_model_checkpoint,
    load_checkpoint,
    get_latest_checkpoint
)
from .dist_utils import (
    init_distributed_mode,
    is_main_process,
    get_rank,
    get_world_size,
    main_process
)
from .storage_utils import CloudPath

# Create logger for this module
logger = logging.getLogger(__name__)


class Runner:
    """Main runner for watermarking system training with GAN losses.

    Attributes:
        config: Root configuration object.
        watermarking_config: Shortcut to ``config.watermarking``.
        discriminator_config: Shortcut to ``config.discriminator``.
        run_config: Shortcut to ``config.run``.
        loss_config: Shortcut to ``config.loss``.
        dataset_config: Shortcut to ``config.dataset``.
        optimizer_config: Shortcut to ``config.optimizer``.
        metrics_config: Shortcut to ``config.metrics``.

        watermarking_system: AudioWatermarking model (possibly DDP-wrapped).
        discriminator: Optional discriminator (possibly DDP-wrapped).

        mel_loss: MelSpectrogramLoss module.
        mel_loss_weight: Weight applied to mel loss.
        waveform_loss: Optional L1Loss for raw waveform comparison.
        waveform_loss_weight: Weight applied to waveform loss.
        stft_loss: Optional MultiScaleSTFTLoss module.
        stft_loss_weight: Weight applied to STFT loss.
        tf_loudness_loss: Optional TFLoudnessLoss module.
        tf_loudness_loss_weight: Weight applied to TF loudness loss.
        localization_loss: Optional LocalizationLoss module.
        localization_loss_weight: Weight applied to localization loss.
        decoding_loss: Optional DecodingLoss module.
        decoding_loss_weight: Weight applied to decoding loss.
        gan_loss: GANLoss (created only when ``use_gan`` is True).
        adv_gen_loss_weight: Weight for generator adversarial loss.
        adv_feat_loss_weight: Weight for feature-matching loss.
        discriminator_update_freq: How often (in iterations) to update the discriminator.

        iteration: Current training iteration.
        epoch: Current epoch counter.
        best_loss: Best validation loss seen so far.
        patience_counter: Consecutive validations without improvement.

        device: Torch device for training.
        use_distributed: Whether DDP is active.
        use_gan: Whether adversarial training is enabled.
        scaler: Optional GradScaler for mixed-precision training.

        loggers: Composite logger (metric, TensorBoard, WandB).
        output_dir: Path to experiment output directory.
        optimizers: Dict of optimizers keyed by component name.
        schedulers: Dict of LR schedulers keyed by component name.
        datasets: Dict of dataset splits (train, valid, test).
        job_id: Unique identifier for this training run.
    """

    def __init__(
        self,
        config: Config,
        watermarking_system: AudioWatermarking,
        discriminator: Optional[nn.Module] = None,
        datasets: Dict[str, Any] = None,
        job_id: str = None
    ):
        self.config = config
        self._watermarking_system = watermarking_system
        self._discriminator = discriminator
        if datasets is None:
            raise ValidationError("datasets parameter is required")
        self.datasets = datasets
        self.job_id = job_id or time.strftime('%Y%m%d_%H%M%S')
        
        self.watermarking_config = config.watermarking
        self.discriminator_config = config.discriminator
        self.run_config = config.run
        self.loss_config = config.loss
        self.dataset_config = config.dataset
        self.optimizer_config = config.optimizer
        self.metrics_config = config.metrics
        
        self.device = self.run_config.device
        self.use_distributed = self.run_config.use_distributed
        self.use_gan = discriminator is not None and self.discriminator_config.use_discriminator
        
        self._watermarking_system.to(self.device)
        if self._discriminator is not None:
            self._discriminator.to(self.device)
        
        if self.use_distributed:
            self._watermarking_system = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._watermarking_system)
            self.watermarking_system = DDP(
                self._watermarking_system,
                device_ids=[self.run_config.gpu] if self.run_config.gpu is not None else None,
                find_unused_parameters=False,
                static_graph=False,  # conditional forward paths prevent static graph
            )
            
            if self._discriminator is not None:
                self._discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._discriminator)
                self.discriminator = DDP(
                    self._discriminator,
                    device_ids=[self.run_config.gpu] if self.run_config.gpu is not None else None,
                    find_unused_parameters=False,
                    static_graph=False,
                )
        else:
            self.watermarking_system = self._watermarking_system
            self.discriminator = self._discriminator

        if is_main_process():
            logger.info("Applying torch.compile for performance optimization")
            try:
                self.watermarking_system.generator = torch.compile(self.watermarking_system.generator)
                self.watermarking_system.detector = torch.compile(self.watermarking_system.detector)
                self.watermarking_system.locator = torch.compile(self.watermarking_system.locator)

                if self.discriminator is not None:
                    self.discriminator = torch.compile(self.discriminator)

                logger.info("torch.compile applied successfully to all models")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        
        self.output_dir = self.run_config.output_dir / self.run_config.experiment_name
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loggers = setup_logger(
            self.output_dir,
            name=f"watermarking_system_{self.job_id}",
            use_tensorboard=self.run_config.tensorboard_enabled,
            use_wandb=self.run_config.wandb_enabled,
            wandb_project=self.run_config.wandb_project
        )
        
        # --- Loss modules ---
        self.mel_loss = MelSpectrogramLoss(**self.loss_config.mel_loss_params)
        self.mel_loss_weight = self.loss_config.mel_loss_weight

        self.waveform_loss = L1Loss(weight=1.0) if self.loss_config.waveform_loss_weight > 0 else None
        self.waveform_loss_weight = self.loss_config.waveform_loss_weight

        self.stft_loss = MultiScaleSTFTLoss(**self.loss_config.stft_loss_params) if self.loss_config.stft_loss_weight > 0 else None
        self.stft_loss_weight = self.loss_config.stft_loss_weight

        self.tf_loudness_loss_weight = self.loss_config.tf_loudness_loss_weight
        self.tf_loudness_loss = TFLoudnessLoss(**self.loss_config.tf_loudness_loss_params) if self.tf_loudness_loss_weight > 0 else None

        self.localization_loss_weight = getattr(self.loss_config, 'localization_loss_weight', 1.0)
        localization_params = getattr(self.loss_config, 'localization_loss_params', {'pos_weight': 1.0, 'neg_weight': 1.0})
        self.localization_loss = LocalizationLoss(
            **localization_params
        ) if self.localization_loss_weight > 0 else None

        self.decoding_loss_weight = getattr(self.loss_config, 'decoding_loss_weight', 1.0)
        self.decoding_loss = DecodingLoss(
            **getattr(self.loss_config, 'decoding_loss_params', {'pos_weight': 1.0, 'neg_weight': 1.0})
        ) if self.decoding_loss_weight > 0 else None

        if self.use_gan and self.loss_config.use_gan_loss:
            self.gan_loss = GANLoss()
            self.adv_gen_loss_weight = self.loss_config.adv_gen_loss_weight
            self.adv_feat_loss_weight = self.loss_config.adv_feat_loss_weight
            self.discriminator_update_freq = self.loss_config.discriminator_update_freq
        


        # --- Training state (must precede checkpoint loading) ---
        self.iteration = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

        self.scaler = GradScaler() if self.run_config.amp else None

        if self.run_config.resume_from:
            self.load_checkpoint(self.run_config.resume_from)

        # --- Optimizers and schedulers ---
        # Uses unwrapped models to avoid DDP attribute-access issues.
        opt_config = get_optimizers_and_schedulers(
            watermarking_system=self._watermarking_system,
            discriminator=self._discriminator,
            lr=self.optimizer_config.learning_rate,
            d_lr=self.optimizer_config.d_learning_rate,
            betas=tuple(self.optimizer_config.betas),
            d_betas=tuple(self.optimizer_config.d_betas),
            eps=self.optimizer_config.eps,
            d_eps=self.optimizer_config.d_eps,
            weight_decay=self.optimizer_config.weight_decay,
            d_weight_decay=self.optimizer_config.d_weight_decay,
            scheduler_type=self.optimizer_config.scheduler_type,
            warmup_steps=self.optimizer_config.warmup_steps,
            max_iterations=self.run_config.max_iterations,
            gamma=self.optimizer_config.gamma,
            min_lr=self.optimizer_config.min_lr,
            cycle_length=self.optimizer_config.cycle_length,
            min_lr_factor=self.optimizer_config.min_lr_factor,
            num_cycles=self.optimizer_config.num_cycles,
            max_lr=self.optimizer_config.max_lr,
            warmup_start_lr=self.optimizer_config.warmup_start_lr,
            start_step=0,
        )

        self.optimizers = opt_config['optimizers']
        self.schedulers = opt_config['schedulers']

        self.setup_data()
    
    def setup_data(self):
        """Create DataLoaders from provided or config-derived datasets."""
        if self.datasets:
            train_dataset = self.datasets.get('train')
            if train_dataset:
                if self.use_distributed:
                    train_sampler = DistributedSampler(
                        train_dataset,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                        shuffle=True
                    )
                    shuffle = False
                else:
                    train_sampler = None
                    shuffle = True
                
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.run_config.batch_size_train,
                    sampler=train_sampler,
                    shuffle=shuffle if train_sampler is None else False,
                    num_workers=self.dataset_config.num_workers,
                    collate_fn=collater,
                    pin_memory=self.dataset_config.pin_memory,
                    drop_last=True
                )
                self.train_sampler = train_sampler
                self.train_loader = IterLoader(self.train_loader, use_distributed=self.use_distributed)

            val_dataset = self.datasets.get('valid')
            if val_dataset:
                if self.use_distributed:
                    val_sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                        shuffle=False
                    )
                else:
                    val_sampler = None
                
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.run_config.batch_size_eval,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=self.dataset_config.num_workers,
                    collate_fn=collater,
                    pin_memory=self.dataset_config.pin_memory,
                    drop_last=False
                )
                self.val_sampler = val_sampler
        else:
            train_dataset = AudioDataset(
                manifest_path=self.dataset_config.train_manifest,
                segment_length=self.dataset_config.segment_length,
                sample_rate=self.watermarking_config.generator.sample_rate,
                normalize=True,
                random_segment=True,
                device="cpu"
            )
            
            if self.use_distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=True
                )
                shuffle = False
            else:
                train_sampler = None
                shuffle = True

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.run_config.batch_size_train,
                sampler=train_sampler,
                shuffle=shuffle if train_sampler is None else False,
                num_workers=self.dataset_config.num_workers,
                collate_fn=collater,
                pin_memory=self.dataset_config.pin_memory,
                drop_last=True
            )
            self.train_sampler = train_sampler
            self.train_loader = IterLoader(self.train_loader, use_distributed=self.use_distributed)

            val_dataset = AudioDataset(
                manifest_path=self.dataset_config.valid_manifest,
                segment_length=self.dataset_config.segment_length,
                sample_rate=self.watermarking_config.generator.sample_rate,
                normalize=True,
                random_segment=False,
                device="cpu"
            )
            
            if self.use_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=False
                )
            else:
                val_sampler = None
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.run_config.batch_size_eval,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                collate_fn=collater,
                pin_memory=self.dataset_config.pin_memory,
                drop_last=False
            )
            self.val_sampler = val_sampler
    
    def train_discriminator_step(self, real_audio: torch.Tensor,
                                fake_audio: torch.Tensor,
                                accumulation_step: int = 0) -> Dict[str, float]:
        """Train discriminator for one gradient-accumulation sub-step."""
        if accumulation_step == 0:
            self.optimizers['discriminator'].zero_grad()

        with autocast(device_type='cuda', enabled=self.run_config.amp):
            real_scores = self.discriminator(real_audio)
            fake_scores = self.discriminator(fake_audio)
            total_d_loss, d_losses = self.gan_loss.discriminator_loss(real_scores, fake_scores)

        total_d_loss = total_d_loss / self.run_config.accum_grad_iters

        if self.scaler:
            self.scaler.scale(total_d_loss).backward()
            if accumulation_step == self.run_config.accum_grad_iters - 1:
                apply_gradient_clipping(
                    self.optimizers['discriminator'],
                    grad_clip_norm=10.0
                )
                self.scaler.step(self.optimizers['discriminator'])
                self.scaler.update()
                self.schedulers['discriminator'].step()
        else:
            total_d_loss.backward()
            if accumulation_step == self.run_config.accum_grad_iters - 1:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.optimizer_config.discriminator_grad_clip_norm
                )
                self.optimizers['discriminator'].step()
                self.schedulers['discriminator'].step()

        if accumulation_step == self.run_config.accum_grad_iters - 1:
            total_norm = sum(
                p.grad.norm()**2 for p in self.discriminator.parameters() 
                if p.grad is not None
            )**0.5 if any(p.grad is not None for p in self.discriminator.parameters()) else 0.0
            grad_norm = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
            d_losses['disc_grad_norm'] = grad_norm
        
        return d_losses
    
    def train_watermarking_step(self, audio_batch: torch.Tensor,
                               accumulation_step: int = 0,
                               bits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Train watermarking system for one gradient-accumulation sub-step.

        All objectives (localization, detection, reconstruction, adversarial)
        are combined into a single backward pass so gradients flow cleanly.

        Returns:
            (reconstructed_signal, watermarked_audio, audio_clean, losses_dict)
        """
        if accumulation_step == 0:
            self.optimizers['watermarking'].zero_grad()

        if bits is None:
            batch_size = audio_batch.shape[0]

            model_pattern_tensor = torch.tensor(self.watermarking_config.model_pattern,
                                               dtype=torch.long, device=self.device)
            model_bits = model_pattern_tensor.unsqueeze(0).expand(batch_size, -1)
            data_bits = torch.randint(0, 2, (batch_size, 16), dtype=torch.long, device=self.device)
            bits = torch.cat([model_bits, data_bits], dim=1)

        loss_dict = {}
        total_loss = 0.0

        with autocast(device_type='cuda', enabled=self.run_config.amp):
            audio_clean = audio_batch.clone()

            # Forward returns a 7-tuple during training:
            # (reconstructed, watermarked, wm_detector, wm_locator, mask, clean_detector, clean_locator)
            if isinstance(self.watermarking_system, DDP):
                outputs = self.watermarking_system.module(audio_clean, bits, phase=self.watermarking_config.train_phase)
            else:
                outputs = self.watermarking_system(audio_clean, bits, phase=self.watermarking_config.train_phase)

            reconstructed_signal = outputs[0]
            watermarked_signal = outputs[1]
            watermarked_detector_output = outputs[2]   # [B, nbits, T]
            watermarked_locator_output = outputs[3]    # [B, 1, T]
            mask = outputs[4]
            clean_detector_output = outputs[5]         # [B, nbits, T]
            clean_locator_output = outputs[6]          # [B, 1, T]

            watermarked = watermarked_signal
            
            if self.localization_loss is not None:
                localization_loss = self.localization_loss(
                    watermarked_locator_output=watermarked_locator_output,
                    ground_truth_mask=mask,
                    clean_locator_output=clean_locator_output,
                )
                weighted_loc = self.localization_loss_weight * localization_loss
                total_loss = total_loss + weighted_loc
                loss_dict['localization_loss'] = localization_loss.item()
            
            if self.decoding_loss is not None:
                decoding_loss = self.decoding_loss(
                    watermarked_detector_output=watermarked_detector_output,
                    ground_truth_mask=mask,
                    ground_truth_message=bits,
                    clean_detector_output=clean_detector_output,
                )
                weighted_dec = self.decoding_loss_weight * decoding_loss
                total_loss = total_loss + weighted_dec
                loss_dict['decoding_loss'] = decoding_loss.item()
            
            # --- Reconstruction losses ---
            mel_loss = self.mel_loss(watermarked, audio_clean)
            weighted_mel = self.mel_loss_weight * mel_loss
            total_loss = total_loss + weighted_mel
            loss_dict['mel_loss'] = mel_loss.item()

            if self.waveform_loss is not None:
                waveform_loss = self.waveform_loss(watermarked, audio_clean)
                weighted_waveform = self.waveform_loss_weight * waveform_loss
                total_loss = total_loss + weighted_waveform
                loss_dict['waveform_loss'] = waveform_loss.item()
            
            if self.stft_loss is not None:
                stft_dict = self.stft_loss(watermarked, audio_clean)
                stft_loss = stft_dict["total"]
                weighted_stft = self.stft_loss_weight * stft_loss
                total_loss = total_loss + weighted_stft
                loss_dict['stft_loss'] = stft_loss.item()

            if self.tf_loudness_loss is not None:
                tf_loudness_loss = self.tf_loudness_loss(watermarked, audio_clean)
                weighted_tf_loudness = self.tf_loudness_loss_weight * tf_loudness_loss
                total_loss = total_loss + weighted_tf_loudness
                loss_dict['tf_loudness_loss'] = tf_loudness_loss.item()
            
            if self.use_gan and self.iteration > 0:
                fake_scores = self.discriminator(reconstructed_signal)
                adv_gen_loss, _ = self.gan_loss.generator_loss(fake_scores)
                weighted_adv = self.adv_gen_loss_weight * adv_gen_loss
                total_loss = total_loss + weighted_adv
                loss_dict['adv_gen_loss'] = adv_gen_loss.item()
                
                if self.adv_feat_loss_weight > 0:
                    with torch.no_grad():
                        real_features = self.discriminator(audio_clean)
                    fake_features = self.discriminator(reconstructed_signal)
                    
                    adv_feat_loss = self.gan_loss.feature_matching_loss(real_features, fake_features)
                    weighted_feat = self.adv_feat_loss_weight * adv_feat_loss
                    total_loss = total_loss + weighted_feat
                    loss_dict['adv_feat_loss'] = adv_feat_loss.item()
            
            loss_dict['total_loss'] = total_loss.item()
            
        # Single backward pass -- discriminator was already updated above,
        # so generator receives fresh adversarial gradients.
        scaled_loss = total_loss / self.run_config.accum_grad_iters

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        def compute_grad_norm(parameters):
            total_norm = sum(
                p.grad.norm()**2 for p in parameters
                if p.grad is not None
            )**0.5 if any(p.grad is not None for p in parameters) else 0.0
            return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

        grad_norm = compute_grad_norm(self.watermarking_system.parameters())
        loss_dict['grad_norm'] = grad_norm
        

        if accumulation_step == self.run_config.accum_grad_iters - 1:
            if self.scaler:
                self.scaler.unscale_(self.optimizers['watermarking'])
                torch.nn.utils.clip_grad_norm_(
                    self.watermarking_system.parameters(),
                    self.optimizer_config.watermarking_grad_clip_norm
                )
                self.scaler.step(self.optimizers['watermarking'])
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.watermarking_system.parameters(),
                    self.optimizer_config.watermarking_grad_clip_norm
                )
                self.optimizers['watermarking'].step()

            self.schedulers['watermarking'].step()
        
        return reconstructed_signal, watermarked, audio_clean, loss_dict
    
    
    def train_iteration(self) -> Dict[str, float]:
        """Run one training iteration (discriminator first, then generator)."""
        self.watermarking_system.train()
        if self.discriminator:
            self.discriminator.train()
        
        all_losses = {}
        accumulated_w_losses = {}
        accumulated_d_losses = {}
        
        for accum_step in range(self.run_config.accum_grad_iters):
            batch = next(self.train_loader)
            audio_batch = batch["raw_wav"].to(self.device)

            batch_size = audio_batch.shape[0]

            model_pattern_tensor = torch.tensor(self.watermarking_config.model_pattern,
                                               dtype=torch.long, device=self.device)
            model_bits = model_pattern_tensor.unsqueeze(0).expand(batch_size, -1)
            data_bits = torch.randint(0, 2, (batch_size, 16), dtype=torch.long, device=self.device)
            bits = torch.cat([model_bits, data_bits], dim=1)

            # Generate detached watermarked samples for discriminator input
            with torch.no_grad():
                audio_clean = audio_batch.clone()
                if isinstance(self.watermarking_system, DDP):
                    outputs = self.watermarking_system.module(audio_clean, bits, phase=self.watermarking_config.train_phase)
                else:
                    outputs = self.watermarking_system(audio_clean, bits, phase=self.watermarking_config.train_phase)
                watermarked_for_disc = outputs[0].detach()

            if (self.use_gan and
                self.iteration > 0 and
                self.iteration % self.discriminator_update_freq == 0):
                d_losses = self.train_discriminator_step(audio_clean, watermarked_for_disc, accum_step)
                for k, v in d_losses.items():
                    if k in accumulated_d_losses:
                        accumulated_d_losses[k] += v
                    else:
                        accumulated_d_losses[k] = v
            
            reconstructed, watermarked, audio_clean_out, w_losses = self.train_watermarking_step(audio_batch, accum_step, bits)
            for k, v in w_losses.items():
                if k in accumulated_w_losses:
                    accumulated_w_losses[k] += v
                else:
                    accumulated_w_losses[k] = v
        
        for k in accumulated_w_losses:
            all_losses[k] = accumulated_w_losses[k] / self.run_config.accum_grad_iters
        for k in accumulated_d_losses:
            all_losses[k] = accumulated_d_losses[k] / self.run_config.accum_grad_iters
        
        all_losses['lr_watermarking'] = self.optimizers['watermarking'].param_groups[0]['lr']
        if self.use_gan:
            all_losses['lr_disc'] = self.optimizers['discriminator'].param_groups[0]['lr']
        
        # Sync losses across ranks at log intervals for consistent monitoring
        if self.use_distributed and self.iteration % self.run_config.log_interval == 0:
            synced_losses = {}
            for key, value in all_losses.items():
                if key not in ['lr_g', 'lr_d']:  # Don't sync learning rates
                    loss_tensor = torch.tensor(value, device=self.device)
                    dist.all_reduce(loss_tensor)
                    synced_losses[key] = loss_tensor.item() / get_world_size()  # Average across GPUs
                else:
                    synced_losses[key] = value
            return synced_losses
        
        return all_losses
    
    def _validate_single_effect(self, effect_name: str = None) -> Dict[str, float]:
        """Run validation for a single effect (or None for default)."""
        val_losses = []
        val_metrics = []
        all_effect_params_used = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.run_config.num_valid_batches:
                    break

                audio_batch = batch["raw_wav"].to(self.device)
                if audio_batch.dim() == 2:
                    audio_batch = audio_batch.unsqueeze(1)
                
                with autocast(device_type='cuda', enabled=self.run_config.amp):
                    audio_clean = audio_batch.clone()

                    batch_size = audio_batch.shape[0]
                    model_pattern_tensor = torch.tensor(self.watermarking_config.model_pattern,
                                                       dtype=torch.long, device=self.device)
                    model_bits = model_pattern_tensor.unsqueeze(0).expand(batch_size, -1)
                    data_bits = torch.randint(0, 2, (batch_size, 16), dtype=torch.long, device=self.device)
                    bits = torch.cat([model_bits, data_bits], dim=1)

                    output = self.watermarking_system(
                        audio_clean, 
                        bits, 
                        phase=self.watermarking_config.valid_phase,
                        specific_effect=effect_name
                    )
                    
                    reconstructed_signal = output['reconstructed_signal']
                    watermarked_signal = output['watermarked_signal']
                    watermarked = watermarked_signal

                    results_dict = output.get('results_dict', {})

                    # Collect effect parameters for scheduler tracking
                    if results_dict and 'attacks' in results_dict:
                        attack_stats = results_dict['attacks'].get('stats', {})
                        if 'effect_params_used' in attack_stats:
                            # This contains the actual parameters used for each effect
                            # Convert list of tuples to dict if needed
                            params_list = attack_stats['effect_params_used']
                            if isinstance(params_list, list) and params_list:
                                # Store all parameters used in this batch
                                for effect, params in params_list:
                                    if effect == effect_name or (effect_name is None and effect == 'identity'):
                                        # Add to our collection of all parameters used
                                        all_effect_params_used.append((effect, params))
                            elif isinstance(params_list, dict):
                                # Already in dict format - convert to tuple format
                                for effect, params in params_list.items():
                                    if effect == effect_name or (effect_name is None and effect == 'identity'):
                                        all_effect_params_used.append((effect, params))
                    
                    if results_dict:
                        effect_key = effect_name if effect_name else 'identity'
                        effect_results = results_dict.get(effect_key, list(results_dict.values())[0])
                        detector_full = effect_results['watermarked_detector_output']  # [B, nbits, T]
                        clean_detector_full = effect_results['clean_detector_output']  # [B, nbits, T]
                    else:
                        detector_full = output['watermarked_detector_output']  # [B, nbits, T]
                        clean_detector_full = output['clean_detector_output']  # [B, nbits, T]
                    
                    positive_logits = detector_full
                    mask = output['ground_truth_mask']

                    total_loss = 0.0
                    batch_metrics = {}
                    
                    # Get locator outputs from results_dict or output
                    if results_dict and effect_key in results_dict:
                        # Get locator output from effect results
                        locator_full = effect_results['locator_output']
                    else:
                        # Get locator output directly from output dict
                        locator_full = output['watermarked_locator_output']

                    clean_locator_output = output.get('clean_locator_output')

                    loc_metrics = compute_localization_metrics(
                        positive=locator_full,
                        negative=clean_locator_output if clean_locator_output is not None else torch.zeros_like(locator_full),
                        target_mask_pos=mask,
                        target_mask_neg=torch.zeros_like(mask) if clean_locator_output is not None else None,
                        threshold=self.watermarking_config.locator.localization_threshold
                    )

                    if 'combined_miou' in loc_metrics:
                        loc_metrics['localization_iou'] = loc_metrics['combined_miou']

                    batch_metrics.update(loc_metrics)

                    negative_logits = clean_detector_full

                    det_metrics = compute_detection_metrics(
                        positive_logits,
                        negative_logits,
                        bits,
                        mask
                    )

                    batch_metrics.update(det_metrics)

                    mel_loss = self.mel_loss(watermarked, audio_clean)
                    total_loss += self.mel_loss_weight * mel_loss
                    batch_metrics['mel_loss'] = mel_loss.item()
                    
                    # Multi-scale STFT loss
                    if self.stft_loss_weight > 0:
                        stft_dict = self.stft_loss(watermarked, audio_clean)
                        stft_loss = stft_dict["total"]
                        total_loss += self.stft_loss_weight * stft_loss
                        batch_metrics['stft_loss'] = stft_loss.item()

                    # TF Loudness loss (disabled in validation)
                    # if self.tf_loudness_loss is not None:
                    #     tf_loudness_loss = self.tf_loudness_loss(watermarked, audio_clean)
                    #     total_loss += self.tf_loudness_loss_weight * tf_loudness_loss
                    #     batch_metrics['tf_loudness_loss'] = tf_loudness_loss.item()


                val_losses.append(total_loss.item())

                audio_metrics_list = []
                if self.metrics_config.use_sisnr and 'sisnr' in self.metrics_config.validation_metrics:
                    audio_metrics_list.append('sisnr')
                if self.metrics_config.use_pesq and 'pesq' in self.metrics_config.validation_metrics:
                    audio_metrics_list.append('pesq')
                if self.metrics_config.use_stoi and 'stoi' in self.metrics_config.validation_metrics:
                    audio_metrics_list.append('stoi')
                
                metrics = {}
                if audio_metrics_list:
                    metrics = compute_audio_reconstruction_metrics(
                        watermarked,
                        audio_batch,
                        sample_rate=self.watermarking_config.generator.sample_rate,
                        metrics_list=audio_metrics_list,
                        pesq_sample_rate=self.metrics_config.pesq_sample_rate,
                        pesq_mode=self.metrics_config.pesq_mode
                    )
                
                metrics.update(batch_metrics)
                val_metrics.append(metrics)
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {}
        if val_metrics:
            for key in val_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in val_metrics])

        if self.use_distributed:
            val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor)
            avg_val_loss = val_loss_tensor.item() / get_world_size()

            for key in list(avg_metrics.keys()):
                metric_tensor = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(metric_tensor)
                avg_metrics[key] = metric_tensor.item() / get_world_size()

        result = {
            'val_loss': avg_val_loss,
            **avg_metrics
        }
        
        # Aggregate parameters used across all batches
        # For now, we'll return the most frequently used parameters
        if all_effect_params_used:
            from collections import Counter
            
            # Log collected parameters for debugging
            logger.debug(f"Collected {len(all_effect_params_used)} parameter sets during validation")
            
            # Count frequency of each parameter combination
            param_counts = Counter()
            for effect, params in all_effect_params_used:
                # Create a hashable representation of the parameters
                if 'frequency_pairs' in params and 'choices' in params['frequency_pairs']:
                    # For bandpass, use the frequency pair as key
                    freq_pair = tuple(params['frequency_pairs']['choices'][0]) if params['frequency_pairs']['choices'] else ()
                    param_key = ('bandpass', freq_pair)
                elif params:
                    # For other effects, use a tuple of sorted items
                    param_key = (effect, tuple(sorted(params.items())))
                else:
                    # For effects with no parameters (like identity)
                    param_key = (effect, ())
                param_counts[param_key] += 1
            
            # Get the most common parameters
            if param_counts:
                most_common = param_counts.most_common(1)[0]
                effect_name, param_tuple = most_common[0]
                
                logger.debug(f"Most common params for {effect_name}: {param_tuple} (used {most_common[1]} times)")
                
                # Reconstruct the parameters dict
                if effect_name == 'bandpass' and param_tuple:
                    # Reconstruct bandpass parameters using correct format
                    freq_pair = list(param_tuple)
                    result['effect_params_used'] = {
                        'frequency_pairs': {
                            'choices': [freq_pair]
                        }
                    }
                elif param_tuple:
                    # Reconstruct other parameters
                    result['effect_params_used'] = dict(param_tuple)
                else:
                    # No parameters (like identity)
                    result['effect_params_used'] = {}
        else:
            logger.debug("No effect parameters collected during validation")
        
        return result
    
    def _validate_single_effect_with_params(
        self,
        effect_name: str,
        specific_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Run validation for one effect with fixed parameters."""
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.run_config.num_valid_batches:
                    break
                
                audio_batch = batch["raw_wav"].to(self.device)
                if audio_batch.dim() == 2:
                    audio_batch = audio_batch.unsqueeze(1)
                
                with autocast(device_type='cuda', enabled=self.run_config.amp):
                    audio_clean = audio_batch.clone()

                    batch_size = audio_batch.shape[0]
                    model_pattern_tensor = torch.tensor(self.watermarking_config.model_pattern,
                                                       dtype=torch.long, device=self.device)
                    model_bits = model_pattern_tensor.unsqueeze(0).expand(batch_size, -1)
                    data_bits = torch.randint(0, 2, (batch_size, 16), dtype=torch.long, device=self.device)
                    bits = torch.cat([model_bits, data_bits], dim=1)

                    output = self.watermarking_system(
                        audio_clean, 
                        bits, 
                        phase=self.watermarking_config.valid_phase,
                        specific_effect=effect_name,
                        specific_params=specific_params if effect_name != 'identity' else {}
                    )
                    
                    reconstructed_signal = output['reconstructed_signal']
                    watermarked_signal = output['watermarked_signal']
                    watermarked = watermarked_signal

                    results_dict = output.get('results_dict', {})

                    if results_dict:
                        effect_key = effect_name
                        effect_results = results_dict.get(effect_key, list(results_dict.values())[0])
                        detector_full = effect_results['watermarked_detector_output']
                        clean_detector_full = effect_results['clean_detector_output']
                    else:
                        detector_full = output['watermarked_detector_output']
                        clean_detector_full = output['clean_detector_output']
                    
                    positive_logits = detector_full
                    mask = output['ground_truth_mask']

                    total_loss = 0.0
                    batch_metrics = {}

                    if results_dict and effect_name in results_dict:
                        locator_full = results_dict[effect_name].get('locator_output', detector_full)
                    else:
                        locator_full = output['watermarked_locator_output']

                    clean_locator_output = output.get('clean_locator_output')

                    loc_metrics = compute_localization_metrics(
                        positive=locator_full,
                        negative=clean_locator_output if clean_locator_output is not None else torch.zeros_like(locator_full),
                        target_mask_pos=mask,
                        target_mask_neg=torch.zeros_like(mask) if clean_locator_output is not None else None,
                        threshold=self.watermarking_config.locator.localization_threshold
                    )

                    if 'combined_miou' in loc_metrics:
                        batch_metrics['localization_iou'] = loc_metrics['combined_miou']

                    batch_metrics.update(loc_metrics)

                    negative_logits = clean_detector_full

                    det_metrics = compute_detection_metrics(
                        positive_logits,
                        negative_logits,  # [B, nbits, T] detector outputs on clean audio
                        bits,             # [B, nbits] original message
                        mask              # [B, 2, T] watermark presence mask
                    )
                    
                    batch_metrics.update(det_metrics)

                    if self.decoding_loss:
                        decoding_loss = self.decoding_loss(
                            watermarked_detector_output=detector_full,
                            ground_truth_mask=mask,
                            ground_truth_message=bits
                        )
                        total_loss += self.decoding_loss_weight * decoding_loss
                        batch_metrics['decoding_loss'] = decoding_loss.item()
                    
                    if self.mel_loss:
                        mel_loss = self.mel_loss(watermarked, audio_clean)
                        total_loss += self.mel_loss_weight * mel_loss
                        batch_metrics['mel_loss'] = mel_loss.item()
                    
                    if self.stft_loss:
                        stft_dict = self.stft_loss(watermarked, audio_clean)
                        stft_loss = stft_dict["total"]
                        total_loss += self.stft_loss_weight * stft_loss
                        batch_metrics['stft_loss'] = stft_loss.item()

                    # TF Loudness loss (disabled in validation)
                    # if self.tf_loudness_loss:
                    #     tf_loudness_loss = self.tf_loudness_loss(watermarked, audio_clean)
                    #     total_loss += self.tf_loudness_loss_weight * tf_loudness_loss
                    #     batch_metrics['tf_loudness_loss'] = tf_loudness_loss.item()

                    batch_metrics['total_loss'] = total_loss.item()
                    
                    val_losses.append(total_loss.item())
                    val_metrics.append(batch_metrics)
        
        # Average metrics across batches
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        avg_metrics = {}
        if val_metrics:
            metric_keys = val_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in val_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
        
        # Synchronize metrics across GPUs if using DDP
        if get_world_size() > 1:
            val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor)
            avg_val_loss = val_loss_tensor.item() / get_world_size()
            
            for key in list(avg_metrics.keys()):
                metric_tensor = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(metric_tensor)
                avg_metrics[key] = metric_tensor.item() / get_world_size()
        
        result = {
            'val_loss': avg_val_loss,
            **avg_metrics,
            'effect_params_used': specific_params
        }
        
        return result
    
    def _get_all_parameter_combinations(self, effect_name: str, effect_params_config: Dict) -> List[Dict[str, Any]]:
        """Enumerate all parameter combinations for a given effect."""
        if effect_name == 'identity':
            return [{}]

        if effect_name == 'bandpass':
            bandpass_config = getattr(effect_params_config, 'bandpass', {})
            freq_pairs_config = bandpass_config.get('frequency_pairs', {})
            freq_pairs = freq_pairs_config.get('choices', [])

            if not freq_pairs:
                logger.warning(f"No frequency pairs found for bandpass effect")
                return [{}]

            return [
                {
                    'frequency_pairs': {
                        'choices': [pair]
                    }
                }
                for pair in freq_pairs
            ]
        
        effect_config = getattr(effect_params_config, effect_name, {})
        all_combos = []
        
        for param_name, param_config in effect_config.items():
            if isinstance(param_config, dict) and 'choices' in param_config:
                choices = param_config['choices']
                for choice in choices:
                    all_combos.append({param_name: choice})
        
        return all_combos if all_combos else [{}]
    
    def validate(self) -> Dict[str, float]:
        """Validate with per-effect and per-parameter localized BER tracking."""
        self.watermarking_system.eval()
        
        effect_config = self.watermarking_config.valid_attacks.effect
        effect_enabled = effect_config.effect_enabled
        
        enabled_effects = [name for name, enabled in effect_enabled.items()
                          if enabled and name != 'identity']
        effects_to_test = ['identity'] + enabled_effects

        per_param_metrics = {}
        per_effect_summary = {}
        all_metrics = {}
        
        effect_params_config = effect_config.effect_params if hasattr(effect_config, 'effect_params') else {}

        total_tests = 0
        for effect_name in effects_to_test:
            param_combos = self._get_all_parameter_combinations(effect_name, effect_params_config)
            total_tests += len(param_combos)
        
        if is_main_process():
            print(f"\n" + "="*70)
            print(f"Starting systematic parameter validation: {total_tests} total tests")
            print("="*70)
        
        test_count = 0
        for effect_name in effects_to_test:
            param_combinations = self._get_all_parameter_combinations(effect_name, effect_params_config)
            
            if is_main_process():
                print(f"\nTesting {effect_name}: {len(param_combinations)} parameter combinations")
            
            effect_results = []
            
            for params in param_combinations:
                test_count += 1
                
                if is_main_process():
                    param_str = str(params) if params else "no params"
                    print(f"  [{test_count}/{total_tests}] Testing {effect_name} with {param_str}")
                
                metrics = self._validate_single_effect_with_params(effect_name, params)

                if 'localized_ber' in metrics:
                    localized_ber = metrics['localized_ber']
                    global_ber = metrics.get('global_ber', localized_ber)  # Fallback to localized if global not available
                    iou = metrics.get('localization_iou', 0.95)
                    
                    param_key = f"{effect_name}_" + "_".join(
                        f"{k}={v}" for k, v in sorted(params.items()) if not k.startswith('_')
                    ) if params else effect_name
                    
                    per_param_metrics[param_key] = {
                        'effect': effect_name,
                        'params': params,
                        'localized_ber': localized_ber,
                        'global_ber': global_ber,
                        'iou': iou
                    }
                    
                    result_dict = {
                        'params': params,
                        'localized_ber': localized_ber,
                        'iou': iou
                    }

                    # Add dual watermark metrics if available
                    if 'model_ber' in metrics:
                        result_dict['model_ber'] = metrics['model_ber']
                    if 'data_ber' in metrics:
                        result_dict['data_ber'] = metrics['data_ber']
                    if 'model_ber_neg' in metrics:
                        result_dict['model_ber_neg'] = metrics['model_ber_neg']
                    if 'data_ber_neg' in metrics:
                        result_dict['data_ber_neg'] = metrics['data_ber_neg']

                    effect_results.append(result_dict)
            
            if effect_results:
                avg_ber = np.mean([r['localized_ber'] for r in effect_results])
                avg_iou = np.mean([r['iou'] for r in effect_results])
                best_result = min(effect_results, key=lambda x: x['localized_ber'])
                worst_result = max(effect_results, key=lambda x: x['localized_ber'])

                summary_dict = {
                    'avg_ber': avg_ber,
                    'avg_iou': avg_iou,
                    'best_ber': best_result['localized_ber'],
                    'best_params': best_result['params'],
                    'worst_ber': worst_result['localized_ber'],
                    'worst_params': worst_result['params'],
                    'num_params_tested': len(effect_results)
                }

                # Add dual watermark metrics if available
                if 'model_ber' in effect_results[0]:
                    summary_dict['avg_model_ber'] = np.mean([r.get('model_ber', 0.5) for r in effect_results])
                    summary_dict['avg_data_ber'] = np.mean([r.get('data_ber', 0.5) for r in effect_results])
                if 'model_ber_neg' in effect_results[0]:
                    summary_dict['avg_model_ber_neg'] = np.mean([r.get('model_ber_neg', 0.5) for r in effect_results])
                    summary_dict['avg_data_ber_neg'] = np.mean([r.get('data_ber_neg', 0.5) for r in effect_results])

                per_effect_summary[effect_name] = summary_dict
                
                # Store average metrics for wandb
                all_metrics[f"{effect_name}_avg_ber"] = avg_ber
                all_metrics[f"{effect_name}_avg_iou"] = avg_iou
                all_metrics[f"{effect_name}_best_ber"] = best_result['localized_ber']
                all_metrics[f"{effect_name}_worst_ber"] = worst_result['localized_ber']
        
        # Update scheduler with ALL parameter combination results
        if isinstance(self.watermarking_system, DDP):
            train_pipeline = self.watermarking_system.module.train_attacks_pipeline
        else:
            train_pipeline = self.watermarking_system.train_attacks_pipeline
        
        if (train_pipeline is not None and 
            hasattr(train_pipeline, 'effect_attack') and
            hasattr(train_pipeline.effect_attack, 'scheduler') and
            train_pipeline.effect_attack.scheduler is not None):
            
            if is_main_process():
                print(f"\nUpdating scheduler with {len(per_param_metrics)} parameter test results")
            
            # Update scheduler with metrics from EACH parameter combination tested
            for param_key, metrics in per_param_metrics.items():
                if metrics['effect'] == 'identity':
                    continue  # Skip baseline
                
                # Update scheduler with this specific parameter combination's performance
                train_pipeline.update_scheduler_metrics(
                    effect_name=metrics['effect'],
                    effect_params=metrics['params'],
                    ber=metrics['localized_ber'],
                    miou=metrics['iou']
                )
            
            if is_main_process():
                logger.debug(f"Updated scheduler with {len(per_param_metrics) - 1} parameter combinations")
            
            # Adapt probabilities based on validation performance
            # This happens less frequently than training (only at validation intervals)
            train_pipeline.adapt_scheduler_probabilities()
        
        # Use localized BER from identity as baseline
        if 'identity' in per_effect_summary:
            all_metrics['localized_ber'] = per_effect_summary['identity']['avg_ber']
            # Add dual watermark metrics if available
            if 'avg_model_ber' in per_effect_summary['identity']:
                all_metrics['model_ber'] = per_effect_summary['identity']['avg_model_ber']
            if 'avg_data_ber' in per_effect_summary['identity']:
                all_metrics['data_ber'] = per_effect_summary['identity']['avg_data_ber']
            if 'avg_model_ber_neg' in per_effect_summary['identity']:
                all_metrics['model_ber_neg'] = per_effect_summary['identity']['avg_model_ber_neg']
            if 'avg_data_ber_neg' in per_effect_summary['identity']:
                all_metrics['data_ber_neg'] = per_effect_summary['identity']['avg_data_ber_neg']

        # Add other metrics from the last effect run (for general metrics)
        last_effect_metrics = self._validate_single_effect(None)
        for key, value in last_effect_metrics.items():
            # Don't overwrite the metrics we've already set
            if key not in ['localized_ber', 'model_ber', 'data_ber', 'model_ber_neg', 'data_ber_neg', 'model_localized_ber', 'data_localized_ber']:
                all_metrics[key] = value
        
        # Log per-effect summary statistics
        if is_main_process():
            print("\n" + "="*70)
            print("Validation Summary - Systematic Parameter Testing")
            print("(Using Localized BER - computed only in watermarked regions)")
            print("="*70)
            for effect_name, summary in per_effect_summary.items():
                print(f"\n{effect_name} ({summary['num_params_tested']} parameter combinations):")
                print(f"  Average Localized BER: {summary['avg_ber']*100:6.2f}% / IoU: {summary['avg_iou']:.3f}")

                # Add dual watermark performance if available
                if 'avg_model_ber' in summary and 'avg_data_ber' in summary:
                    print(f"  Model Watermark BER (pos): {summary['avg_model_ber']*100:6.2f}%")
                    print(f"  Data Watermark BER (pos):  {summary['avg_data_ber']*100:6.2f}%")
                if 'avg_model_ber_neg' in summary and 'avg_data_ber_neg' in summary:
                    print(f"  Model Watermark BER (neg): {summary['avg_model_ber_neg']*100:6.2f}%")
                    print(f"  Data Watermark BER (neg):  {summary['avg_data_ber_neg']*100:6.2f}%")

                print(f"  Best Localized BER:    {summary['best_ber']*100:6.2f}% with params: {summary['best_params']}")
                print(f"  Worst Localized BER:   {summary['worst_ber']*100:6.2f}% with params: {summary['worst_params']}")
            print("="*70 + "\n")
        
        # Log training scheduler statistics during validation
        # This shows how the adaptive learning is progressing
        if is_main_process():
            # Get the training pipeline's scheduler (not validation)
            if isinstance(self.watermarking_system, DDP):
                train_pipeline = self.watermarking_system.module.train_attacks_pipeline
            else:
                train_pipeline = self.watermarking_system.train_attacks_pipeline
            
            if (train_pipeline is not None and 
                hasattr(train_pipeline, 'effect_attack') and
                hasattr(train_pipeline.effect_attack, 'scheduler') and
                train_pipeline.effect_attack.scheduler is not None):
                
                scheduler_stats = train_pipeline.get_scheduler_statistics()
                if scheduler_stats:
                    print("\n" + "="*70)
                    print(f"Training Scheduler Statistics (at validation, iteration {self.iteration})")
                    print("(BER values are localized - computed only in watermarked regions)")
                    print("-"*70)
                    
                    for effect_name, stats in scheduler_stats.items():
                        if isinstance(stats, dict) and 'ema_ber' in stats and stats['ema_ber'] is not None:
                            # Get values with proper None handling
                            ema_ber = stats['ema_ber']
                            ema_miou = stats.get('ema_miou')
                            probability = stats.get('probability')
                            
                            # Format with defaults for None values
                            ber_str = f"{ema_ber:.4f}" if ema_ber is not None else "N/A"
                            miou_str = f"{ema_miou:.4f}" if ema_miou is not None else "N/A"
                            prob_str = f"{probability:.2%}" if probability is not None else "N/A"
                            
                            print(f"{effect_name:15s} / EMA Localized BER: {ber_str} / "
                                  f"EMA IoU: {miou_str} / "
                                  f"Prob: {prob_str}")
                            
                            # Show frequency pair statistics for bandpass
                            if effect_name == 'bandpass' and 'frequency_pair_stats' in stats:
                                print("\n  Frequency Pair Success Rates:")
                                for freq_pair, pair_stats in stats['frequency_pair_stats'].items():
                                    success_rate = pair_stats.get('success_rate')
                                    samples = pair_stats.get('samples', 0)
                                    rate_str = f"{success_rate:.2%}" if success_rate is not None else "N/A"
                                    print(f"    {freq_pair:20s} / Success: {rate_str} "
                                          f"(n={samples})")
                    
                    print("="*70 + "\n")
        
        return all_metrics
    
    def log_iteration(self, losses: Dict[str, float]):
        """Log training losses to all configured backends."""
        self.loggers.metric_logger.update(**losses)

        if hasattr(self.loggers, 'tensorboard'):
            self.loggers.tensorboard.add_metrics(
                losses,
                self.iteration,
                prefix='train'
            )
        
        if hasattr(self.loggers, 'wandb'):
            self.loggers.wandb.log(
                {f'train/{k}': v for k, v in losses.items()},
                step=self.iteration
            )

        if self.iteration % self.run_config.log_interval == 0:
            metric_str = self.loggers.metric_logger.__str__()
            print(f"Iteration {self.iteration}: {metric_str}")
    
    @main_process
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint (and best-model copy when ``is_best``)."""
        checkpoint_path = self.output_dir / f"checkpoint_{self.iteration:08d}.pth"

        objects_to_save = {
            'optimizer_watermarking': self.optimizers['watermarking'].state_dict(),
            'scheduler_watermarking': self.schedulers['watermarking'].state_dict(),
            'iteration': self.iteration,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config.model_dump()
        }
        
        if self.use_gan and self.discriminator is not None:
            discriminator_to_save = self.unwrap_model(self.discriminator) if self.use_distributed else self.discriminator
            objects_to_save['discriminator'] = discriminator_to_save.state_dict()
            objects_to_save['optimizer_d'] = self.optimizers['discriminator'].state_dict()
            objects_to_save['scheduler_d'] = self.schedulers['discriminator'].state_dict()
        
        if self.scaler is not None:
            objects_to_save['scaler'] = self.scaler.state_dict()
        
        save_model_checkpoint(
            model=self.watermarking_system,
            save_path=checkpoint_path,
            use_distributed=self.run_config.use_distributed,
            drop_untrained_params=False,
            **objects_to_save
        )
        
        config_json_path = checkpoint_path.with_suffix('.json')
        self.config.save_model_json(config_json_path)
        
        latest_path = self.output_dir / "checkpoint_latest.pth"
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(checkpoint_path.name)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(checkpoint_path, latest_path)
        
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pth"
            save_model_checkpoint(
                model=self.watermarking_system,
                save_path=best_path,
                use_distributed=self.run_config.use_distributed,
                drop_untrained_params=False,
                iteration=self.iteration,
                epoch=self.epoch,
                config=self.config.model_dump()
            )
            best_config_json_path = best_path.with_suffix('.json')
            self.config.save_model_json(best_config_json_path)
        
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last_n: int = 5):
        """Keep only the last ``keep_last_n`` numbered checkpoints."""
        checkpoints = []
        for ckpt in self.output_dir.glob("checkpoint_*.pth"):
            if ckpt.name in ["checkpoint_latest.pth", "checkpoint_best.pth"]:
                continue
            if ckpt.stem.replace("checkpoint_", "").isdigit():
                checkpoints.append(ckpt)
        
        checkpoints.sort()
        
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
                json_path = ckpt.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
    
    def unwrap_model(self, model):
        """Unwrap model from DDP if needed."""
        if self.use_distributed:
            return model.module
        return model
    
    def compute_gradient_norm(self) -> torch.Tensor:
        """Compute L2 norm of gradients for monitoring."""
        total_norm = sum(
            p.grad.data.norm(p=2).pow(2)
            for p in self.watermarking_system.parameters()
            if p.grad is not None
        )
        if isinstance(total_norm, torch.Tensor):
            return total_norm.sqrt()
        return torch.tensor(0.0, device=self.device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint and restore all optimizer/scheduler state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        def _force_lr(optimizer: torch.optim.Optimizer, lr: float):
            """Ensure optimizer param groups use requested learning rate."""
            for group in optimizer.param_groups:
                group['lr'] = lr

        def _sync_scheduler_lr(scheduler, lr: float):
            """Keep scheduler base/current LR aligned with optimizer override."""
            if hasattr(scheduler, 'base_lrs'):
                scheduler.base_lrs = [lr for _ in scheduler.base_lrs]
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr = [lr for _ in scheduler._last_lr]
            elif hasattr(scheduler, 'get_last_lr'):
                # Some schedulers lazily create _last_lr; calling once seeds it.
                current = scheduler.get_last_lr()
                scheduler._last_lr = [lr for _ in current]
        
        watermarking_to_load = self.unwrap_model(self.watermarking_system) if self.use_distributed else self.watermarking_system

        if 'watermarking_system_state_dict' in checkpoint:
            watermarking_to_load.load_state_dict(checkpoint['watermarking_system_state_dict'], strict=False)
            print("Loaded watermarking system state from 'watermarking_system_state_dict'")
        else:
            print("Warning: No 'watermarking_system_state_dict' found in checkpoint")
        
        if 'optimizer_watermarking' in checkpoint:
            self.optimizers['watermarking'].load_state_dict(checkpoint['optimizer_watermarking'])
            _force_lr(self.optimizers['watermarking'], self.optimizer_config.learning_rate)

        if 'scheduler_watermarking' in checkpoint:
            try:
                self.schedulers['watermarking'].load_state_dict(checkpoint['scheduler_watermarking'])
                print("Loaded scheduler state successfully")
                _sync_scheduler_lr(self.schedulers['watermarking'], self.optimizer_config.learning_rate)
            except KeyError as e:
                if 'lr_lambdas' in str(e):
                    print(f"Warning: Scheduler type mismatch detected")
                    print(f"Previous: constant scheduler | Current: cosine_cyclic scheduler")
                    print(f"Resetting scheduler to correct step for cosine cyclic scheduling")
                    from .optims import get_cosine_cyclic_schedule_with_warmup

                    old_scheduler = self.schedulers['watermarking']
                    new_scheduler = get_cosine_cyclic_schedule_with_warmup(
                        optimizer=self.optimizers['watermarking'],
                        num_warmup_steps=self.optimizer_config.warmup_steps,
                        cycle_length=self.optimizer_config.cycle_length,
                        min_lr_factor=self.optimizer_config.min_lr_factor,
                        num_cycles=self.optimizer_config.num_cycles,
                        max_lr=self.optimizer_config.max_lr,
                        warmup_start_lr=self.optimizer_config.warmup_start_lr,
                        start_step=self.iteration
                    )
                    self.schedulers['watermarking'] = new_scheduler
                    _sync_scheduler_lr(self.schedulers['watermarking'], self.optimizer_config.learning_rate)
                    print(f"Created new cosine cyclic scheduler starting from step {self.iteration}")
                else:
                    raise
        
        if self.use_gan and self.discriminator is not None:
            if 'discriminator' in checkpoint:
                discriminator_to_load = self.unwrap_model(self.discriminator) if self.use_distributed else self.discriminator
                discriminator_to_load.load_state_dict(checkpoint['discriminator'])
            if 'optimizer_d' in checkpoint and self.optimizers.get('discriminator'):
                self.optimizers['discriminator'].load_state_dict(checkpoint['optimizer_d'])
                _force_lr(self.optimizers['discriminator'], self.optimizer_config.d_learning_rate)
            if 'scheduler_d' in checkpoint and self.schedulers.get('discriminator'):
                try:
                    self.schedulers['discriminator'].load_state_dict(checkpoint['scheduler_d'])
                    print("Loaded discriminator scheduler state successfully")
                    _sync_scheduler_lr(self.schedulers['discriminator'], self.optimizer_config.d_learning_rate)
                except KeyError as e:
                    if 'lr_lambdas' in str(e):
                        print(f"Warning: Discriminator scheduler type mismatch detected")
                        print(f"Previous: constant scheduler | Current: cosine_cyclic scheduler")
                        from .optims import get_cosine_cyclic_schedule_with_warmup

                        new_scheduler = get_cosine_cyclic_schedule_with_warmup(
                            optimizer=self.optimizers['discriminator'],
                            num_warmup_steps=self.optimizer_config.warmup_steps,
                            cycle_length=self.optimizer_config.cycle_length,
                            min_lr_factor=self.optimizer_config.min_lr_factor,
                            num_cycles=self.optimizer_config.num_cycles,
                            max_lr=self.optimizer_config.d_learning_rate,
                            warmup_start_lr=self.optimizer_config.warmup_start_lr,
                            start_step=self.iteration
                        )
                        self.schedulers['discriminator'] = new_scheduler
                        _sync_scheduler_lr(self.schedulers['discriminator'], self.optimizer_config.d_learning_rate)
                        print(f"Created new discriminator cosine cyclic scheduler starting from step {self.iteration}")
                    else:
                        raise
        
        if self.scaler is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.iteration = checkpoint.get('iteration', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Resumed from iteration {self.iteration}")
    
    @main_process
    def generate_samples(self):
        """Generate and log audio samples to TensorBoard/WandB."""
        self.watermarking_system.eval()
        
        if self.run_config.val_idx:
            val_samples = []
            val_dataset = self.datasets.get('valid', self.datasets.get('test'))
            for idx in self.run_config.val_idx[:self.run_config.num_audio_samples]:
                if idx < len(val_dataset):
                    sample = val_dataset[idx]
                    val_samples.append(sample["raw_wav"])
            
            if val_samples:
                val_batch = torch.stack(val_samples).to(self.device)
                if val_batch.dim() == 2:
                    val_batch = val_batch.unsqueeze(1)
            else:
                batch = next(iter(self.val_loader))
                val_batch = batch["raw_wav"].to(self.device)
                if val_batch.dim() == 2:
                    val_batch = val_batch.unsqueeze(1)
                val_batch = val_batch[:self.run_config.num_audio_samples]
        else:
            batch = next(iter(self.val_loader))
            val_batch = batch["raw_wav"].to(self.device)
            if val_batch.dim() == 2:
                val_batch = val_batch.unsqueeze(1)
            val_batch = val_batch[:self.run_config.num_audio_samples]
        
        with torch.no_grad():
            batch_size = val_batch.shape[0]

            model_pattern_tensor = torch.tensor(self.watermarking_config.model_pattern,
                                               dtype=torch.long, device=self.device)
            model_bits = model_pattern_tensor.unsqueeze(0).expand(batch_size, -1)
            data_bits = torch.randint(0, 2, (batch_size, 16), dtype=torch.long, device=self.device)
            bits = torch.cat([model_bits, data_bits], dim=1)

            output = self.watermarking_system(val_batch, bits, phase=self.watermarking_config.valid_phase)

            reconstructed_signal = output['reconstructed_signal']
            watermarked_signal = output['watermarked_signal']
            reconstructed = watermarked_signal

        if hasattr(self.loggers, 'tensorboard'):
            for i in range(val_batch.shape[0]):
                self.loggers.tensorboard.add_watermark_samples(
                    val_batch[i],
                    reconstructed[i],
                    self.iteration,
                    self.watermarking_config.generator.sample_rate,
                    prefix=f"samples_{i}"
                )
        
        if hasattr(self.loggers, 'wandb'):
            for i in range(val_batch.shape[0]):
                self.loggers.wandb.log_watermark_comparison(
                    val_batch[i],
                    reconstructed[i],
                    self.iteration,
                    self.watermarking_config.generator.sample_rate
                )
    
    def train(self):
        """Entry point for training - delegates to run()."""
        self.run()
    
    def run(self):
        """Run the main training loop."""
        if is_main_process():
            print(f"Starting training from iteration {self.iteration}")
            print(f"Training for {self.run_config.max_iterations} iterations")
        
        while self.iteration < self.run_config.max_iterations:
            losses = self.train_iteration()
            self.log_iteration(losses)
            
            if self.iteration % self.run_config.validation_interval == 0:
                val_metrics = self.validate()
                
                if hasattr(self.loggers, 'tensorboard'):
                    self.loggers.tensorboard.add_metrics(
                        val_metrics,
                        self.iteration,
                        prefix='val'
                    )
                
                if hasattr(self.loggers, 'wandb'):
                    self.loggers.wandb.log(
                        {f'val/{k}': v for k, v in val_metrics.items()},
                        step=self.iteration
                    )
                
                if is_main_process():
                    print(f"Validation at iteration {self.iteration}: {val_metrics}")
                
                val_loss = val_metrics.get(self.run_config.checkpoint_best_metric.replace('/', '_'), val_metrics['val_loss'])
                is_best = False

                if self.run_config.checkpoint_best_mode == "min":
                    is_best = val_loss < (self.best_loss - self.run_config.early_stop_min_delta)
                else:
                    is_best = val_loss > (self.best_loss + self.run_config.early_stop_min_delta)
                
                if is_best:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.run_config.early_stop_patience:
                    if is_main_process():
                        print(f"Early stopping triggered after {self.patience_counter} validations without improvement")
                    break
            
            if self.iteration % self.run_config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            if (self.run_config.save_iters and
                self.iteration in self.run_config.save_iters):
                self.save_checkpoint()
            
            sample_interval = self.run_config.sample_freq or self.run_config.validation_interval
            if (self.run_config.save_audio_samples and 
                self.iteration % sample_interval == 0):
                self.generate_samples()
            
            self.iteration += 1

            if self.use_distributed:
                dist.barrier()
        
        if is_main_process():
            print("Training completed!")
        
        self.save_checkpoint()

        if hasattr(self.loggers, 'tensorboard'):
            self.loggers.tensorboard.close()
        if hasattr(self.loggers, 'wandb'):
            self.loggers.wandb.finish()
