# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Audio watermark locator: detects WHERE watermarks exist in the time domain."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.seanet import SEANetEncoder
from ..config import LocatorConfig

logger = logging.getLogger("AudioAuth")


class Locator(nn.Module):
    """Temporal watermark presence detector.

    Outputs per-sample logits indicating watermark presence at each time step
    (BCE-based, single channel). Does NOT extract message bits — that is the
    Detector's job.

    Attributes:
        config: Locator configuration.
        ratios: Downsampling ratios matching the generator encoder.
        dimension: Latent representation channels.
        output_dim: Intermediate channels before final projection.
        sample_rate: Expected audio sample rate in Hz.
        localization_threshold: Sigmoid threshold for presence decisions.
        hop_length: Total downsampling factor (product of ratios).
        output_channels: Output channels (1 for BCE localization).
        stride: Transposed conv stride, equals hop_length.
        kernel_size: Transposed conv kernel, equals hop_length.
        encoder: SEANet encoder shared architecture.
        reverse_convolution: Transposed conv restoring temporal resolution.
        last_layer: 1x1 conv projecting to presence logits.

    Args:
        config: Locator configuration containing model parameters.
    """

    def __init__(self, config: LocatorConfig) -> None:
        """Initialize the watermark locator.

        Args:
            config: Locator configuration containing model parameters.
        """
        super().__init__()

        try:
            if config is None:
                raise ValueError("LocatorConfig is required - no fallback to defaults allowed")
            if not isinstance(config, LocatorConfig):
                raise TypeError(f"Expected LocatorConfig, got {type(config)}")
            self.config = config
            logger.info("Initializing Locator with config")

            required_attrs = ['encoder_rates', 'latent_dim', 'output_dim',
                            'sample_rate', 'localization_threshold', 'output_channels']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    raise ValueError(f"LocatorConfig missing required attribute: {attr}")

            self.ratios: Tuple[int, ...] = tuple(config.encoder_rates)
            self.dimension: int = config.latent_dim
            self.output_dim: int = config.output_dim
            self.sample_rate: int = config.sample_rate
            self.localization_threshold: float = config.localization_threshold
            self.hop_length: int = int(np.prod(self.ratios))
            self.output_channels: int = config.output_channels

            self.stride: int = int(np.prod(self.ratios))
            self.kernel_size: int = int(np.prod(self.ratios))

            # msg_dimension is unused by the locator but required by SEANetEncoder's interface
            self.encoder = SEANetEncoder(
                channels=config.encoder_channels,
                dimension=self.dimension,
                msg_dimension=32,
                n_filters=config.encoder_dim,
                n_fft_base=config.encoder_n_fft_base,
                n_residual_layers=config.encoder_residual_layers,
                ratios=self.ratios,
                activation=config.encoder_activation,
                activation_params=config.encoder_activation_params,
                norm=config.encoder_norm,
                norm_params=config.encoder_norm_params,
                kernel_size=config.encoder_kernel_size,
                last_kernel_size=config.encoder_last_kernel_size,
                residual_kernel_size=config.encoder_residual_kernel_size,
                dilation_base=config.encoder_dilation_base,
                skip=config.encoder_skip,
                causal=config.encoder_causal,
                pad_mode=config.encoder_pad_mode,
                act_all=config.encoder_act_all,
                expansion=config.encoder_expansion,
                groups=config.encoder_groups,
                l2norm=config.encoder_l2norm,
                bias=config.encoder_bias,
                spec=config.encoder_spec,
                spec_compression=config.encoder_spec_compression,
                spec_learnable=config.encoder_spec_learnable,
                res_scale=config.encoder_res_scale,
                wav_std=config.encoder_wav_std,
                spec_means=config.encoder_spec_means,
                spec_stds=config.encoder_spec_stds,
                zero_init=config.encoder_zero_init,
                inout_norm=config.encoder_inout_norm,
                embedding_dim=64,
                embedding_layers=2,
                freq_bands=4
            )

            self.reverse_convolution = torch.nn.ConvTranspose1d(
                in_channels=self.dimension,
                out_channels=self.output_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0
            )
            nn.init.kaiming_normal_(self.reverse_convolution.weight, nonlinearity='relu')
            if self.reverse_convolution.bias is not None:
                nn.init.constant_(self.reverse_convolution.bias, 0)

            # Single-channel output: logits for BCE-based watermark presence
            self.last_layer = nn.Conv1d(self.output_dim, self.output_channels, 1)
            nn.init.kaiming_normal_(self.last_layer.weight, nonlinearity='relu')
            if self.last_layer.bias is not None:
                nn.init.constant_(self.last_layer.bias, 0)

        except Exception as e:
            logger.error(f"Failed to initialize Locator: {str(e)}", exc_info=True)
            raise RuntimeError(f"Locator initialization failed: {str(e)}")

    def preprocess(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int]
    ) -> Tuple[int, torch.Tensor]:
        """Pad audio to nearest multiple of hop_length.

        Args:
            audio_data: Audio tensor of shape (B, C, T).
            sample_rate: Sample rate in Hz. If None, uses model's sample rate.

        Returns:
            Tuple of (original_length, padded_audio).
        """
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate

            assert sample_rate == self.sample_rate, \
                f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"

            length: int = audio_data.shape[-1]
            right_pad: int = math.ceil(length / self.hop_length) * self.hop_length - length
            audio_data = nn.functional.pad(audio_data, (0, right_pad))

            return length, audio_data

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio preprocessing failed: {str(e)}")

    def localize(
        self,
        audio_data: torch.Tensor,
        orig_nframes: int
    ) -> torch.Tensor:
        """Produce per-sample watermark presence logits.

        Args:
            audio_data: Audio tensor of shape (B, 1, T).
            orig_nframes: Original frame count before padding.

        Returns:
            Presence logits of shape (B, 1, T). Positive = watermark likely present.
        """
        try:
            z = self.encoder(audio_data, None)
            z = self.reverse_convolution(z)
            z = z[:, :, :orig_nframes]
            result = self.last_layer(z)
            return result

        except Exception as e:
            logger.error(f"Error in localize: {str(e)}", exc_info=True)
            raise RuntimeError(f"Localization failed: {str(e)}")

    def forward(
        self,
        audio_signal: Union[torch.Tensor, Any]
    ) -> torch.Tensor:
        """Forward pass: encode audio and produce presence logits.

        Args:
            audio_signal: Audio tensor (B, 1, T) or object with 'audio_data' attr.

        Returns:
            Presence logits of shape (B, 1, T).
        """
        try:
            if hasattr(audio_signal, 'audio_data'):
                length = audio_signal.audio_data.shape[-1]
                audio_data = audio_signal.audio_data
            else:
                length = audio_signal.shape[-1]
                audio_data = audio_signal

            result = self.localize(audio_data, length)
            return result

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"Forward pass failed: {str(e)}")
