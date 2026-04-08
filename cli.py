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


# Encode and decode commands removed - project now focuses on watermarking instead of codec functionality


if __name__ == "__main__":
    audioauth()