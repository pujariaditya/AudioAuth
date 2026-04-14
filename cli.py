# Copyright (c) 2025 Aditya Pujari
# Licensed under the MIT License. See LICENSE file for details.

"""Command-line interface for AudioAuth.

Provides commands for training audio watermarking models.
"""

# Fix OpenMP fork issue with DataLoader workers - must be done before torch import
import os
import multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from pathlib import Path

import click
import torch

from train import main as train_fn
from infer import run_inference as infer_fn


def common_options(f):
    """Common options decorator for CLI commands.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function with common CLI options
    """
    f = click.option("--cfg-path", required=True, type=Path, help="Path to configuration file")(f)
    f = click.option(
        "--options",
        default=[],
        multiple=True,
        type=str,
        help="Override fields in the config. A list of key=value pairs",
    )(f)
    return f


@click.group()
def audioauth():
    """AudioAuth - Audio Watermarking Verification Tool."""
    pass


@audioauth.command()
@common_options
def train(cfg_path, options):
    """Train an AudioAuth model from a configuration file.
    
    Args:
        cfg_path: Path to YAML configuration file
        options: List of config overrides as key=value pairs
    """
    train_fn(cfg_path=cfg_path, options=options)


@audioauth.command()
@click.option("--config", "cfg_path", required=True, type=Path, help="Path to YAML config file")
@click.option("--checkpoint", required=True, type=Path, help="Path to .pth checkpoint")
@click.option("--manifest", default=None, type=Path, help="Path to JSONL manifest (defaults to config's test_manifest)")
@click.option("--output-dir", default="outputs/inference", type=Path, help="Output directory")
@click.option("--num-samples", default=10, type=int, help="Number of samples to process")
@click.option("--device", default="cuda", type=str, help="Device (cuda or cpu)")
@click.option("--max-duration", default=10.0, type=float, help="Max audio duration in seconds")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.pass_context
def encode(ctx, cfg_path, checkpoint, manifest, output_dir, num_samples, device, max_duration, seed):
    """Generate watermarked audio from a trained checkpoint."""
    if manifest is None:
        from AudioAuth.config import Config
        config = Config.from_yaml(cfg_path)
        manifest = Path(config.dataset.test_manifest)

    ctx.invoke(
        infer_fn,
        config_path=cfg_path,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        output_dir=output_dir,
        num_samples=num_samples,
        device=device,
        max_duration=max_duration,
        seed=seed,
    )


if __name__ == "__main__":
    audioauth()