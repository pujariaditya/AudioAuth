"""Discriminator modules for AudioAuth.

Implements discriminators for the AudioAuth framework.
Implements Multi-Period, Multi-Scale, and Multi-Resolution discriminators for GAN training.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
import torchaudio.functional as AF

from .layers import WNConv1d as _WNConv1d

logger = logging.getLogger(__name__)

# Frequency bands for Multi-Resolution Discriminator (normalized frequencies)
BANDS: List[Tuple[float, float]] = [
    (0.0, 0.1),    # Low frequencies
    (0.1, 0.25),   # Low-mid frequencies
    (0.25, 0.5),   # Mid frequencies
    (0.5, 0.75),   # High-mid frequencies
    (0.75, 1.0)    # High frequencies
]

def WNConv1d(*args, **kwargs) -> Union[nn.Module, nn.Sequential]:
    """Weight normalized 1D convolution with optional activation.
    
    Special version for discriminator that adds LeakyReLU activation.
    
    Args:
        *args: Positional arguments to pass to WNConv1d.
        **kwargs: Keyword arguments to pass to WNConv1d.
            Special keyword 'act' (bool): Whether to add activation. Default True.
    
    Returns:
        nn.Module or nn.Sequential: Convolution layer with optional activation.
    
    Raises:
        ValueError: If invalid arguments are provided to the convolution layer.
    """
    try:
        act: bool = kwargs.pop("act", True)
        conv = _WNConv1d(*args, **kwargs)
        if not act:
            return conv
        return nn.Sequential(conv, nn.LeakyReLU(0.1))
    except Exception as e:
        logger.error(f"Error creating WNConv1d layer: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to create WNConv1d layer: {str(e)}")


def WNConv2d(*args, **kwargs) -> Union[nn.Module, nn.Sequential]:
    """Weight normalized 2D convolution with optional activation.
    
    Args:
        *args: Positional arguments to pass to Conv2d.
        **kwargs: Keyword arguments to pass to Conv2d.
            Special keyword 'act' (bool): Whether to add activation. Default True.
    
    Returns:
        nn.Module or nn.Sequential: Convolution layer with optional activation.
    
    Raises:
        ValueError: If invalid arguments are provided to the convolution layer.
    """
    try:
        act: bool = kwargs.pop("act", True)
        conv = weight_norm(nn.Conv2d(*args, **kwargs))
        if not act:
            return conv
        return nn.Sequential(conv, nn.LeakyReLU(0.1))
    except Exception as e:
        logger.error(f"Error creating WNConv2d layer: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to create WNConv2d layer: {str(e)}")

class MPD(nn.Module):
    """Multi-Period Discriminator.
    
    Processes audio by reshaping it into a 2D representation based on a period
    parameter, allowing the discriminator to focus on periodic patterns in audio.
    """
    
    def __init__(self, period: int) -> None:
        """Initialize Multi-Period Discriminator.
        
        Args:
            period: Period parameter for reshaping the audio signal.
                    Must be a positive integer.
        
        Raises:
            ValueError: If period is not a positive integer.
        """
        super().__init__()
        
        # Validate input
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"Period must be a positive integer, got {period}")
            
        self.period = period

        try:
            self.convs = nn.ModuleList([
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ])
            self.conv_post = WNConv2d(
                1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
            )
            
            logger.debug(f"Initialized MPD with period={period}")
            
        except Exception as e:
            logger.error(f"Error initializing MPD layers: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize MPD: {str(e)}")

    def pad_to_period(self, x: torch.Tensor) -> torch.Tensor:
        """Pad tensor to be divisible by period.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
        
        Returns:
            torch.Tensor: Padded tensor with time dimension divisible by period.
        """
        try:
            time_dim = x.shape[-1]
            padding_needed = self.period - (time_dim % self.period)
            
            # Only pad if necessary
            if padding_needed != self.period:
                x = F.pad(x, (0, padding_needed), mode="reflect")
                
            return x
            
        except Exception as e:
            logger.error(f"Error padding tensor: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to pad tensor: {str(e)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of Multi-Period Discriminator.
        
        Args:
            x: Input tensor of shape (batch, 1, time).
        
        Returns:
            List[torch.Tensor]: Feature maps from each layer.
        
        Raises:
            RuntimeError: If forward pass fails.
        """
        try:
            feature_maps: List[torch.Tensor] = []

            x = self.pad_to_period(x)
            # Reshape to 2D so convolutions see periodic structure
            x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

            for layer in self.convs:
                x = layer(x)
                feature_maps.append(x)

            x = self.conv_post(x)
            feature_maps.append(x)
            
            return feature_maps
            
        except Exception as e:
            logger.error(f"Error in MPD forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"MPD forward pass failed: {str(e)}")


class MSD(nn.Module):
    """Multi-Scale Discriminator.
    
    Uses proper anti-aliased downsampling via torchaudio's resample function
    to avoid artifacts that occur with simple average pooling.
    """
    
    def __init__(self, rate: int = 1, sample_rate: int = 16000) -> None:
        """Initialize Multi-Scale Discriminator.
        
        Args:
            rate: Downsampling factor. Must be >= 1.
            sample_rate: Original sample rate of audio in Hz.
        
        Raises:
            ValueError: If rate < 1 or sample_rate <= 0.
        """
        super().__init__()
        
        # Validate inputs
        if rate < 1:
            raise ValueError(f"Rate must be >= 1, got {rate}")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
            
        self.rate = rate
        self.sample_rate = sample_rate
        
        try:
            self.convs = nn.ModuleList([
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ])
            self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
            
            logger.debug(f"Initialized MSD with rate={rate}, sample_rate={sample_rate}")
            
        except Exception as e:
            logger.error(f"Error initializing MSD layers: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize MSD: {str(e)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of Multi-Scale Discriminator.
        
        Args:
            x: Input tensor of shape (batch, 1, time).
        
        Returns:
            List[torch.Tensor]: Feature maps from each layer.
        
        Raises:
            RuntimeError: If forward pass fails.
        """
        try:
            # Anti-aliased downsampling avoids aliasing artifacts from naive pooling
            if self.rate > 1:
                target_sr = self.sample_rate // self.rate
                x = x.squeeze(1)
                x = AF.resample(
                    x,
                    orig_freq=self.sample_rate,
                    new_freq=target_sr,
                    rolloff=0.99,
                    lowpass_filter_width=6
                )
                x = x.unsqueeze(1)

            feature_maps: List[torch.Tensor] = []

            for layer in self.convs:
                x = layer(x)
                feature_maps.append(x)

            x = self.conv_post(x)
            feature_maps.append(x)
            
            return feature_maps
            
        except Exception as e:
            logger.error(f"Error in MSD forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"MSD forward pass failed: {str(e)}")


class MRD(nn.Module):
    """Multi-Resolution Discriminator.
    
    Operates on spectrogram representations of audio, analyzing different
    frequency bands independently before combining results.
    """
    
    def __init__(
        self, 
        window_length: int = 512,
        hop_factor: float = 0.25,
        sample_rate: int = 16000,
        bands: Optional[List[Tuple[float, float]]] = None
    ) -> None:
        """Initialize Multi-Resolution Discriminator.
        
        Args:
            window_length: Window size for STFT computation.
            hop_factor: Hop length as fraction of window length (0 < hop_factor <= 1).
            sample_rate: Sample rate of audio in Hz.
            bands: List of frequency band tuples (low, high) as normalized frequencies.
                   If None, uses default BANDS.
        
        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__()
        
        # Validate inputs
        if window_length <= 0:
            raise ValueError(f"Window length must be positive, got {window_length}")
        if not 0 < hop_factor <= 1:
            raise ValueError(f"Hop factor must be in (0, 1], got {hop_factor}")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
            
        self.window_length = window_length
        self.hop_length = int(window_length * hop_factor)
        self.sample_rate = sample_rate
        
        # Use default frequency bands if not specified
        if bands is None:
            bands = BANDS
        self.bands = bands
        
        try:
            n_fft = window_length // 2 + 1
            self.band_indices = [
                (int(band[0] * n_fft), int(band[1] * n_fft)) 
                for band in bands
            ]
            
            channel_depth = 32

            def create_conv_stack() -> nn.ModuleList:
                """Create a convolutional stack for processing one frequency band."""
                return nn.ModuleList([
                    WNConv2d(2, channel_depth, (3, 9), (1, 1), padding=(1, 4)),
                    WNConv2d(channel_depth, channel_depth, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channel_depth, channel_depth, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channel_depth, channel_depth, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channel_depth, channel_depth, (3, 3), (1, 1), padding=(1, 1)),
                ])
            
            self.band_convs = nn.ModuleList([
                create_conv_stack() for _ in range(len(self.bands))
            ])
            
            self.conv_post = WNConv2d(
                channel_depth, 1, (3, 3), (1, 1), padding=(1, 1), act=False
            )
            
            logger.debug(f"Initialized MRD with {len(bands)} frequency bands")
            
        except Exception as e:
            logger.error(f"Error initializing MRD layers: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize MRD: {str(e)}")

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute STFT and split into frequency bands.
        
        Args:
            x: Input tensor of shape (batch, 1, time).
        
        Returns:
            List[torch.Tensor]: STFT representations for each frequency band.
        
        Raises:
            RuntimeError: If STFT computation fails.
        """
        try:
            x = x.squeeze(1)
            x_stft = torch.stft(
                x,
                n_fft=self.window_length,
                hop_length=self.hop_length,
                win_length=self.window_length,
                window=torch.hann_window(self.window_length, device=x.device),
                return_complex=True,
                center=True,  # Center padding for better edge handling
            )
            
            x_stft = torch.view_as_real(x_stft)
            # (batch, freq, time, 2) -> (batch, 2, time, freq) for 2D conv
            x_stft = rearrange(x_stft, "b f t c -> b c t f")

            x_bands = [
                x_stft[..., band_idx[0]:band_idx[1]] 
                for band_idx in self.band_indices
            ]
            
            return x_bands
            
        except Exception as e:
            logger.error(f"Error computing spectrogram: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to compute spectrogram: {str(e)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of Multi-Resolution Discriminator.
        
        Args:
            x: Input tensor of shape (batch, 1, time).
        
        Returns:
            List[torch.Tensor]: Feature maps from each layer.
        
        Raises:
            RuntimeError: If forward pass fails.
        """
        try:
            x_bands = self.spectrogram(x)
            feature_maps: List[torch.Tensor] = []

            processed_bands: List[torch.Tensor] = []
            for band, conv_stack in zip(x_bands, self.band_convs):
                for layer in conv_stack:
                    band = layer(band)
                    feature_maps.append(band)
                processed_bands.append(band)

            x = torch.cat(processed_bands, dim=-1)
            x = self.conv_post(x)
            feature_maps.append(x)
            
            return feature_maps
            
        except Exception as e:
            logger.error(f"Error in MRD forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"MRD forward pass failed: {str(e)}")

class Discriminator(nn.Module):
    """Main discriminator class combining MPD, MSD, and MRD.
    
    Combines multiple discriminator types to capture different aspects of audio:
    - MPD: Captures periodic patterns
    - MSD: Analyzes at multiple time scales
    - MRD: Examines frequency domain representations
    """
    
    def __init__(
        self,
        rates: List[int] = None,
        periods: List[int] = None,
        fft_sizes: List[int] = None,
        sample_rate: int = 16000,
        bands: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialize combined discriminator.

        Args:
            rates: Downsampling factors for MSD. Empty list disables MSD.
                   Default is empty list.
            periods: Periods for MPD. Default is [2, 3, 5, 7, 11].
            fft_sizes: Window sizes for MRD. Default is [2048, 1024, 512].
            sample_rate: Sample rate of audio in Hz. Default is 16000.
            bands: Frequency bands for MRD. Default is BANDS.
        
        Raises:
            ValueError: If invalid parameters are provided.
        """
        super().__init__()
        
        # Set default values
        if rates is None:
            rates = []
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512]
        if bands is None:
            bands = BANDS
        
        # Validate inputs
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        
        try:
            discriminator_modules: List[nn.Module] = []

            for period in periods:
                discriminator_modules.append(MPD(period))
                logger.debug(f"Added MPD with period={period}")
            
            for rate in rates:
                discriminator_modules.append(MSD(rate, sample_rate=sample_rate))
                logger.debug(f"Added MSD with rate={rate}")
            
            for fft_size in fft_sizes:
                discriminator_modules.append(
                    MRD(fft_size, sample_rate=sample_rate, bands=bands)
                )
                logger.debug(f"Added MRD with fft_size={fft_size}")
            
            self.discriminators = nn.ModuleList(discriminator_modules)
            
            logger.info(
                f"Initialized Discriminator with {len(self.discriminators)} sub-discriminators: "
                f"{len(periods)} MPD, {len(rates)} MSD, {len(fft_sizes)} MRD"
            )
            
        except Exception as e:
            logger.error(f"Error initializing Discriminator: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Discriminator: {str(e)}")

    def preprocess(self, y: torch.Tensor) -> torch.Tensor:
        """Preprocess audio before discrimination.
        
        Applies DC offset removal and peak normalization.
        
        Args:
            y: Input audio tensor of shape (batch, channels, time).
        
        Returns:
            torch.Tensor: Preprocessed audio tensor.
        """
        try:
            y = y - y.mean(dim=-1, keepdim=True)
            # Scale to 80% of peak to leave headroom and prevent clipping
            max_amplitude = y.abs().max(dim=-1, keepdim=True)[0]
            y = 0.8 * y / (max_amplitude + 1e-9)
            
            return y
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to preprocess audio: {str(e)}")

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Forward pass through all discriminators.
        
        Args:
            x: Input audio tensor of shape (batch, 1, time).
        
        Returns:
            List[List[torch.Tensor]]: Feature maps from all discriminators.
                                      Outer list corresponds to discriminators,
                                      inner lists contain feature maps per discriminator.
        
        Raises:
            RuntimeError: If forward pass fails.
        """
        try:
            x = self.preprocess(x)
            all_feature_maps: List[List[torch.Tensor]] = []

            for discriminator in self.discriminators:
                feature_maps = discriminator(x)
                all_feature_maps.append(feature_maps)
            
            return all_feature_maps
            
        except Exception as e:
            logger.error(f"Error in Discriminator forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"Discriminator forward pass failed: {str(e)}")