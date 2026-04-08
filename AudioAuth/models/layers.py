"""Weight-normalized layers and Snake activation for AudioAuth."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    """Weight-normalized 1D convolution. Wraps nn.Conv1d with weight_norm."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalized 1D transposed convolution. Wraps nn.ConvTranspose1d with weight_norm."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    """Snake activation: x + (1/alpha) * sin^2(alpha * x). JIT-compiled for ~1.4x speedup."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Snake activation for 1D inputs: x + (1/alpha) * sin^2(alpha * x).

    Learnable per-channel alpha controls oscillation frequency. Effective for audio
    due to periodic inductive bias. See https://arxiv.org/abs/2006.08195.

    Attributes:
        alpha: Learnable frequency parameter, shape (1, channels, 1).

    Args:
        channels: Number of channels (each gets its own alpha).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Snake activation.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Activated tensor of shape (B, C, T).
        """
        return snake(x, self.alpha)
