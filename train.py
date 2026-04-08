# Copyright (c) 2025 Aditya Pujari
# Licensed under the MIT License. See LICENSE file for details.

"""Training script for AudioAuth.

Audio watermarking training pipeline with distributed training support.
"""

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from AudioAuth.config import Config
from AudioAuth.dataset import AudioDataset
from AudioAuth.dist_utils import get_rank, init_distributed_mode
from AudioAuth.logger import setup_logger
from AudioAuth.utils import load_models, print_model_summary
from AudioAuth.runner import Runner


def now_as_str() -> str:
    """Get current time as string for job ID.
    
    Returns:
        str: Timestamp string in format YYYYMMDD_HHMMSS
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_seeds(config: Config) -> None:
    """Set random seeds for reproducibility.
    
    Seeds are offset by rank to ensure different initialization across processes
    while maintaining reproducibility.
    
    Args:
        config: Configuration object containing seed and cudnn settings
    """
    seed = config.run.seed + get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cudnn.benchmark = config.run.cudnn_benchmark
    cudnn.deterministic = config.run.cudnn_deterministic


def main(cfg_path: str | Path, options: List[str] = []) -> None:
    """Main training function for AudioAuth model.
    
    Sets up distributed training, loads configuration, initializes models,
    creates datasets, and starts the training loop.
    
    Args:
        cfg_path: Path to YAML configuration file containing model, training,
            and dataset parameters
        options: List of CLI override options in key=value format
            (e.g., ["run.batch_size_train=64", "model.n_codebooks=16"])
    
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    # Enable anomaly detection to help debug gradient computation issues
    torch.autograd.set_detect_anomaly(True)
    
    # Generate job ID before distributed init to ensure all ranks have the same ID
    job_id = now_as_str()
    
    # Load configuration with CLI overrides
    cfg = Config.from_sources(yaml_file=Path(cfg_path), cli_args=options)
    
    # Initialize distributed training
    init_distributed_mode(cfg.run)
    
    # Update job ID and experiment name
    if cfg.run.job_id is None:
        cfg.run.job_id = job_id
    if cfg.run.experiment_name is None:
        cfg.run.experiment_name = f"audioauth_{job_id}"
    
    # Define output directory for all ranks
    output_dir = cfg.run.output_dir / cfg.run.experiment_name
    
    # Pretty print config and save only on rank 0
    if get_rank() == 0:
        cfg.pretty_print()
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_to_yaml(output_dir / "config.yaml")
    
    # Setup seeds and logger
    setup_seeds(cfg)
    setup_logger(
        output_dir=output_dir,
        use_tensorboard=cfg.run.tensorboard_enabled,
        use_wandb=cfg.run.wandb_enabled,
        wandb_project=cfg.run.wandb_project
    )
    
    # Load models
    watermarking_system, discriminator = load_models(
        cfg,
        checkpoint_path=cfg.run.resume_from,
        device=cfg.run.device
    )
    
    # Print model summary on rank 0
    if get_rank() == 0:
        print_model_summary(watermarking_system, discriminator)
    
    # Calculate segment lengths based on duration or use default
    train_segment_length = int(cfg.dataset.train_duration * cfg.watermarking.generator.sample_rate) if cfg.dataset.train_duration else cfg.dataset.segment_length
    valid_segment_length = int(cfg.dataset.valid_duration * cfg.watermarking.generator.sample_rate) if cfg.dataset.valid_duration else cfg.dataset.segment_length
    test_segment_length = int(cfg.dataset.test_duration * cfg.watermarking.generator.sample_rate) if cfg.dataset.test_duration else cfg.dataset.segment_length
    
    # Create datasets
    datasets = {
        "train": AudioDataset(
            manifest_path=cfg.dataset.train_manifest,
            segment_length=train_segment_length,
            sample_rate=cfg.watermarking.generator.sample_rate,
            normalize=True,
            random_segment=True,
            device="cpu"  # Load on CPU, move to GPU in training
        ),
        "valid": AudioDataset(
            manifest_path=cfg.dataset.valid_manifest,
            segment_length=valid_segment_length,
            sample_rate=cfg.watermarking.generator.sample_rate,
            normalize=True,
            random_segment=False,  # Use first segment for validation
            device="cpu"
        ),
        "test": AudioDataset(
            manifest_path=cfg.dataset.test_manifest,
            segment_length=test_segment_length,
            sample_rate=cfg.watermarking.generator.sample_rate,
            normalize=True,
            random_segment=False,  # Use first segment for test
            device="cpu"
        )
    }
    
    # Create runner and start training
    runner = Runner(
        config=cfg,
        watermarking_system=watermarking_system,
        discriminator=discriminator,
        datasets=datasets,
        job_id=cfg.run.job_id
    )
    
    # Start training
    runner.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AudioAuth")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "options",
        nargs="*",
        help="Additional configuration options as key=value pairs"
    )
    
    args = parser.parse_args()
    
    # Run training
    main(args.config, args.options)