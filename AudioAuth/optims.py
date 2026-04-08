"""Optimizer and learning rate scheduler utilities for AudioAuth.

Provides optimizer creation with parameter grouping and various scheduling strategies
for both generator and discriminator in GAN-based watermark training.
"""

import math
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from pytorch_optimizer import AdamP
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.distributed.optim import ZeroRedundancyOptimizer



@dataclass
class OptimizerConfig:
    """Configuration for optimizer creation."""
    name: str = "adamw"  # Support AdamW and AdamP
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.8, 0.99)  # Default from AudioAuth
    weight_decay: float = 0.0  # No weight decay (matching AudioAuth)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1000.0  # Gradient clipping by norm
    grad_clip_value: Optional[float] = None  # Gradient clipping by value


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: str = "exponential"  # exponential, cosine, linear, constant, cosine_cyclic
    warmup_steps: int = 0
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    gamma: float = 0.999996  # For exponential
    T_max: Optional[int] = None  # For cosine
    linear_end_lr: float = 0.0  # For linear

    # Cosine cyclic parameters
    cycle_length: int = 5000  # Steps per cycle for cosine_cyclic
    min_lr_factor: float = 0.5  # Minimum LR as fraction of max_lr for cosine_cyclic
    num_cycles: Optional[int] = None  # Total number of cycles for cosine_cyclic
    max_lr: float = 1.0e-3  # Maximum LR for cosine_cyclic


def get_parameter_groups(model: nn.Module, weight_decay: float = 0.0,
                        skip_list: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different weight decay settings.
    
    Following the standard practice:
    - No weight decay for biases and normalization parameters
    - Regular weight decay for other parameters
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay value
        skip_list: Additional parameter names to skip weight decay
        
    Returns:
        List of parameter groups
    """
    decay = []
    no_decay = []
    
    if skip_list is None:
        skip_list = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if (len(param.shape) == 1 or  # Bias and norm parameters
            name.endswith(".bias") or
            name.endswith(".b") or
            "ln" in name or
            "norm" in name or
            any(skip_name in name for skip_name in skip_list)):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


def create_optimizer(model: nn.Module, config: OptimizerConfig, use_zero: bool = False) -> torch.optim.Optimizer:
    """
    Create optimizer matching AudioAuth approach.

    Args:
        model: Model to optimize
        config: Optimizer configuration
        use_zero: Whether to use ZeroRedundancyOptimizer for distributed training

    Returns:
        Configured optimizer (AdamW or AdamP)
    """
    optimizer_class = AdamP if config.name == "adamp" else AdamW

    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=optimizer_class,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    return optimizer


def create_optimizer_g(model: nn.Module, lr: float = 1e-4, betas: Tuple[float, float] = (0.8, 0.99),
                      weight_decay: float = 0.0, eps: float = 1e-8, use_zero: bool = False) -> torch.optim.Optimizer:
    """Create optimizer for generator (audio tokenizer)."""
    config = OptimizerConfig(
        name="adamw",
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        grad_clip_norm=1000.0  # Higher clip for generator (matching AudioAuth)
    )
    return create_optimizer(model, config, use_zero=use_zero)


def create_optimizer_d(model: nn.Module, lr: float = 1e-4, betas: Tuple[float, float] = (0.8, 0.99),
                      weight_decay: float = 0.0, eps: float = 1e-8, use_zero: bool = False,
                      optimizer: str = "adamw") -> torch.optim.Optimizer:
    """Create optimizer for discriminator."""
    config = OptimizerConfig(
        name=optimizer,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        grad_clip_norm=10.0  # Lower clip for discriminator (matching AudioAuth)
    )
    return create_optimizer(model, config, use_zero=use_zero)


class ExponentialLRScheduler(_LRScheduler):
    """Exponential learning rate decay scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, gamma: float = 0.999996,
                 last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * (self.gamma ** self.last_epoch)
                for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup scheduler that can be combined with other schedulers."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 warmup_start_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class LinearWarmupExponentialScheduler(_LRScheduler):
    """Combined linear warmup + exponential decay scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 gamma: float = 0.999996, warmup_start_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Exponential decay after warmup
            steps_after_warmup = self.last_epoch - self.warmup_steps
            return [base_lr * (self.gamma ** steps_after_warmup)
                    for base_lr in self.base_lrs]


class LinearWarmupCosineScheduler(_LRScheduler):
    """Combined linear warmup + cosine annealing scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 T_max: int, min_lr: float = 1e-6, warmup_start_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.T_max = T_max
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            steps_after_warmup = self.last_epoch - self.warmup_steps
            progress = steps_after_warmup / (self.T_max - self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) *
                    0.5 * (1.0 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


def create_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig,
                    max_iterations: Optional[int] = None, start_step: int = 0) -> _LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        max_iterations: Maximum training iterations (for cosine scheduler)
        start_step: Starting step number (for resuming training)

    Returns:
        Configured scheduler
    """
    if config.name == "exponential":
        if config.warmup_steps > 0:
            scheduler = LinearWarmupExponentialScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                gamma=config.gamma,
                warmup_start_lr=config.warmup_start_lr
            )
        else:
            scheduler = ExponentialLRScheduler(optimizer, gamma=config.gamma)
    
    elif config.name == "cosine":
        T_max = config.T_max or max_iterations
        if T_max is None:
            raise ValueError("T_max or max_iterations must be provided for cosine scheduler")
        
        if config.warmup_steps > 0:
            scheduler = LinearWarmupCosineScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                T_max=T_max,
                min_lr=config.min_lr,
                warmup_start_lr=config.warmup_start_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=config.min_lr
            )
    
    elif config.name == "linear":
        if config.warmup_steps > 0:
            # Linear warmup followed by linear decay
            lambda_lr = lambda step: (
                config.warmup_start_lr / optimizer.param_groups[0]['lr'] + 
                (1 - config.warmup_start_lr / optimizer.param_groups[0]['lr']) * step / config.warmup_steps
                if step < config.warmup_steps
                else 1.0 - (1.0 - config.linear_end_lr / optimizer.param_groups[0]['lr']) * 
                (step - config.warmup_steps) / (max_iterations - config.warmup_steps)
            )
        else:
            # Just linear decay
            lambda_lr = lambda step: (
                1.0 - (1.0 - config.linear_end_lr / optimizer.param_groups[0]['lr']) * 
                step / max_iterations
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    elif config.name == "constant":
        # Constant learning rate (with optional warmup)
        if config.warmup_steps > 0:
            scheduler = LinearWarmupScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                warmup_start_lr=config.warmup_start_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    elif config.name == "cosine_cyclic":
        # Cosine cyclic scheduler with warmup
        scheduler = get_cosine_cyclic_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            cycle_length=config.cycle_length,
            min_lr_factor=config.min_lr_factor,
            num_cycles=config.num_cycles,
            max_lr=config.max_lr,
            warmup_start_lr=config.warmup_start_lr,
            start_step=start_step
        )

    else:
        raise ValueError(f"Unknown scheduler: {config.name}")

    return scheduler


def get_optimizers_and_schedulers(
    watermarking_system: nn.Module,
    discriminator: Optional[nn.Module] = None,
    lr: float = 1e-4,
    d_lr: float = 1e-4,
    betas: Tuple[float, float] = (0.8, 0.99),
    d_betas: Tuple[float, float] = (0.8, 0.99),
    eps: float = 1e-8,
    d_eps: float = 1e-8,
    weight_decay: float = 0.0,
    d_weight_decay: float = 0.0,
    scheduler_type: str = "exponential",
    warmup_steps: int = 0,
    max_iterations: Optional[int] = None,
    gamma: float = 0.999996,
    min_lr: float = 1e-6,
    # Cosine cyclic parameters
    cycle_length: int = 5000,
    min_lr_factor: float = 0.1,
    num_cycles: Optional[int] = None,
    max_lr: float = 1e-5,
    warmup_start_lr: float = 1e-6,
    use_zero: bool = False,
    start_step: int = 0
) -> Dict[str, Any]:
    """
    Get optimizers and schedulers for training with unified optimizer
    for the watermarking system and separate optimizer for discriminator.
    
    Args:
        watermarking_system: AudioWatermarking containing generator, detector, locator
        discriminator: Optional discriminator model for GAN training
        lr: Learning rate for watermarking system
        d_lr: Discriminator learning rate
        betas: Adam betas for watermarking system
        d_betas: Discriminator Adam betas
        eps: Adam epsilon for watermarking system
        d_eps: Discriminator Adam epsilon
        weight_decay: Weight decay for watermarking system
        d_weight_decay: Discriminator weight decay
        scheduler_type: Type of scheduler to use
        warmup_steps: Warmup steps
        max_iterations: Maximum iterations
        gamma: Exponential decay rate
        min_lr: Minimum learning rate
        cycle_length: Steps per cycle for cosine_cyclic scheduler
        min_lr_factor: Minimum LR as fraction of max_lr for cosine_cyclic
        num_cycles: Total number of cycles for cosine_cyclic (None = infinite)
        max_lr: Maximum LR for cosine_cyclic
        warmup_start_lr: Starting learning rate for warmup phase
        use_zero: Whether to use ZeroRedundancyOptimizer for distributed training
        start_step: Starting step number (for resuming training)

    Returns:
        Dictionary with ``'optimizers'`` and ``'schedulers'`` sub-dicts
    """
    if use_zero:
        optimizer_watermarking = ZeroRedundancyOptimizer(
            watermarking_system.parameters(),
            optimizer_class=AdamW,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    else:
        optimizer_watermarking = AdamW(
            watermarking_system.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    optimizers = {
        'watermarking': optimizer_watermarking
    }
    
    if discriminator is not None:
        optimizer_d = create_optimizer_d(
            discriminator, 
            lr=d_lr, 
            betas=d_betas, 
            weight_decay=d_weight_decay, 
            use_zero=use_zero
        )
        optimizers['discriminator'] = optimizer_d
    
    scheduler_config = SchedulerConfig(
        name=scheduler_type,
        gamma=gamma,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        min_lr=min_lr,
        # Cosine cyclic parameters
        cycle_length=cycle_length,
        min_lr_factor=min_lr_factor,
        num_cycles=num_cycles,
        max_lr=max_lr
    )
    
    scheduler_watermarking = create_scheduler(optimizer_watermarking, scheduler_config, max_iterations, start_step)

    schedulers = {
        'watermarking': scheduler_watermarking
    }

    if discriminator is not None:
        scheduler_d = create_scheduler(optimizer_d, scheduler_config, max_iterations, start_step)
        schedulers['discriminator'] = scheduler_d
    
    return {
        'optimizers': optimizers,
        'schedulers': schedulers
    }


def get_cosine_cyclic_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    cycle_length=5000,
    min_lr_factor=0.1,
    num_cycles=None,
    max_lr=1e-5,
    warmup_start_lr=1e-7,
    start_step=0
):
    """
    Create a schedule with a learning rate that:
    1. Linearly warms up from warmup_start_lr to max_lr during warmup
    2. Follows cosine cycles between max_lr and min_lr after warmup
    3. Restarts to max_lr at the beginning of each cycle

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for the warmup phase
        cycle_length: Number of steps in each cosine cycle
        min_lr_factor: Minimum LR as a fraction of max_lr
        num_cycles: Total number of cycles (if None, continues indefinitely)
        max_lr: Maximum learning rate (peak of each cycle)
        warmup_start_lr: Starting learning rate for warmup phase
        start_step: Starting step number (for resuming training)

    Returns:
        LambdaLR scheduler with cosine cyclic learning rate
    """
    min_lr = max_lr * min_lr_factor
    warmup_start_factor = warmup_start_lr / max_lr

    def lr_lambda(current_step):
        actual_step = current_step + start_step

        if actual_step < num_warmup_steps:
            # Linear interpolation from warmup_start_lr to max_lr
            return warmup_start_factor + (1.0 - warmup_start_factor) * (actual_step / max(1, num_warmup_steps))

        # Cosine cycling phase
        steps_after_warmup = actual_step - num_warmup_steps

        # Check if we've exceeded the specified number of cycles
        if num_cycles is not None:
            total_cycle_steps = num_cycles * cycle_length
            if steps_after_warmup >= total_cycle_steps:
                return min_lr_factor  # Stay at min_lr after all cycles

        # Calculate current cycle progress
        cycle_step = steps_after_warmup % cycle_length
        cycle_progress = cycle_step / cycle_length  # 0 to 1 within current cycle

        # Cosine annealing within cycle
        # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
        # When progress=0: lr = max_lr
        # When progress=1: lr = min_lr
        cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_progress))

        # Scale between min_lr_factor and 1.0 (since we return a multiplier of max_lr)
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def apply_gradient_clipping(optimizer: torch.optim.Optimizer,
                          grad_clip_norm: Optional[float] = None,
                          grad_clip_value: Optional[float] = None):
    """
    Apply gradient clipping to optimizer parameters.

    Args:
        optimizer: Optimizer with parameters to clip
        grad_clip_norm: Maximum gradient norm (L2)
        grad_clip_value: Maximum gradient value (element-wise)
    """
    if grad_clip_norm is not None:
        params = []
        for group in optimizer.param_groups:
            params.extend(group['params'])
        torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)

    if grad_clip_value is not None:
        params = []
        for group in optimizer.param_groups:
            params.extend(group['params'])
        torch.nn.utils.clip_grad_value_(params, grad_clip_value)