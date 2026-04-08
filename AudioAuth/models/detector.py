# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Audio watermark detector: extracts embedded message bits from watermarked audio."""

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.seanet import SEANetEncoder
from ..config import DetectorConfig

logger = logging.getLogger("AudioAuth")


class Detector(nn.Module):
    """Extracts embedded message bits from watermarked audio signals.

    Works with Locator for temporal presence detection. The architecture mirrors
    the generator's encoder, followed by a transposed conv to restore temporal
    resolution and a 1x1 conv to project to per-bit logits.

    Attributes:
        config: Detector configuration.
        nbits: Total watermark bits (model_bits + data_bits).
        model_bits: Bits reserved for model identification.
        data_bits: Bits reserved for payload data.
        ratios: Downsampling ratios matching the generator encoder.
        dimension: Latent representation channels.
        output_dim: Intermediate channels before final projection.
        sample_rate: Expected audio sample rate in Hz.
        detection_threshold: Sigmoid threshold for binarizing detected bits.
        hop_length: Total downsampling factor (product of ratios).
        stride: Transposed conv stride, equals hop_length to match encoder downsampling.
        kernel_size: Transposed conv kernel, equals hop_length for exact upsampling.
        encoder: SEANet encoder shared architecture.
        reverse_convolution: Transposed conv restoring temporal resolution.
        decoder: 1x1 conv projecting to per-bit logits.

    Args:
        config: Detector configuration containing nbits and other parameters.
    """

    def __init__(self, config: DetectorConfig) -> None:
        """Initialize the watermark detector.

        Args:
            config: Detector configuration containing model parameters.
        """
        super().__init__()

        try:
            if config is None:
                raise ValueError("DetectorConfig is required - no fallback to defaults allowed")
            if not isinstance(config, DetectorConfig):
                raise TypeError(f"Expected DetectorConfig, got {type(config)}")
            self.config = config
            logger.info("Initializing Detector with config")

            required_attrs = ['nbits', 'encoder_rates', 'latent_dim', 'output_dim',
                            'sample_rate', 'detection_threshold']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    raise ValueError(f"DetectorConfig missing required attribute: {attr}")

            self.nbits: int = config.nbits
            self.model_bits: int = config.model_bits
            self.data_bits: int = config.data_bits
            self.ratios: Tuple[int, ...] = tuple(config.encoder_rates)
            self.dimension: int = config.latent_dim
            self.output_dim: int = config.output_dim
            self.sample_rate: int = config.sample_rate
            self.detection_threshold: float = config.detection_threshold

            logger.info(f"Dual decoding: {self.model_bits} model bits + {self.data_bits} data bits")
            self.hop_length: int = int(np.prod(self.ratios))

            # Stride and kernel must equal hop_length so the transposed conv
            # exactly reverses the encoder's temporal downsampling.
            self.stride: int = int(np.prod(self.ratios))
            self.kernel_size: int = int(np.prod(self.ratios))

            self.encoder = SEANetEncoder(
                channels=config.encoder_channels,
                dimension=self.dimension,
                msg_dimension=self.nbits,
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
                freq_bands=4,
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

            self.decoder = nn.Conv1d(self.output_dim, self.nbits, 1)
            nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='relu')
            if self.decoder.bias is not None:
                nn.init.constant_(self.decoder.bias, 0)

        except Exception as e:
            logger.error(f"Failed to initialize Detector: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detector initialization failed: {str(e)}")

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

    def decode(
        self,
        audio_data: torch.Tensor,
        orig_nframes: int
    ) -> torch.Tensor:
        """Extract per-bit logits from audio at each time step.

        Args:
            audio_data: Audio tensor of shape (B, 1, T).
            orig_nframes: Original frame count before padding.

        Returns:
            Per-bit logits of shape (B, nbits, T).
        """
        try:
            z = self.encoder(audio_data, None)
            z = self.reverse_convolution(z)
            z = z[:, :, :orig_nframes]
            result = self.decoder(z)
            return result

        except Exception as e:
            logger.error(f"Error in decode: {str(e)}", exc_info=True)
            raise RuntimeError(f"Decoding failed: {str(e)}")

    def postprocess(
        self,
        result: torch.Tensor,
        detection_threshold: Optional[float] = None
    ) -> torch.Tensor:
        """Average logits over time and binarize to extract message.

        Args:
            result: Raw detection output of shape (B, nbits, T).
            detection_threshold: Sigmoid threshold for binarization.
                If None, uses self.detection_threshold.

        Returns:
            Extracted binary message of shape (B, nbits).
        """
        if detection_threshold is None:
            detection_threshold = self.detection_threshold

        decoded_message = result.mean(dim=-1)
        message = torch.sigmoid(decoded_message)
        message = torch.gt(message, detection_threshold).int()

        return message

    def postprocess_with_mask(
        self,
        result: torch.Tensor,
        mask: torch.Tensor,
        detection_threshold: Optional[float] = None
    ) -> torch.Tensor:
        """Extract message using locator mask for weighted averaging.

        Focuses bit extraction on regions where the locator detects
        watermark presence, improving accuracy on partially-watermarked audio.

        Args:
            result: Raw detection output of shape (B, nbits, T).
            mask: Locator presence mask of shape (B, 1, T).
            detection_threshold: Sigmoid threshold for binarization.
                If None, uses self.detection_threshold.

        Returns:
            Extracted binary message of shape (B, nbits).
        """
        if detection_threshold is None:
            detection_threshold = self.detection_threshold

        bit_predictions = result
        mask_expanded = mask.expand(-1, self.nbits, -1)
        weighted_bits = bit_predictions * mask_expanded
        mask_sum = mask.sum(dim=-1).clamp(min=1e-6)
        decoded_message = weighted_bits.sum(dim=-1) / mask_sum

        message = torch.sigmoid(decoded_message)
        message = torch.gt(message, detection_threshold).int()

        return message

    def forward(
        self,
        audio_signal: Union[torch.Tensor, Any]
    ) -> torch.Tensor:
        """Forward pass: encode audio and extract per-bit logits.

        Args:
            audio_signal: Audio tensor (B, 1, T) or object with 'audio_data' attr.

        Returns:
            Detection output of shape (B, nbits, T).
        """
        try:
            if hasattr(audio_signal, 'audio_data'):
                length = audio_signal.audio_data.shape[-1]
                audio_data = audio_signal.audio_data
            else:
                length = audio_signal.shape[-1]
                audio_data = audio_signal

            result = self.decode(audio_data, length)
            return result

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def detect(
        self,
        audio_signal: Union[torch.Tensor, Any],
        verbose: bool = False
    ) -> torch.Tensor:
        """Extract embedded watermark message from audio (inference API).

        Args:
            audio_signal: Audio input to extract watermark from.
            verbose: If True, logs extraction results.

        Returns:
            Extracted binary message of shape (B, nbits).
        """
        try:
            with torch.no_grad():
                result = self(audio_signal)
                message = self.postprocess(result)

                if verbose:
                    model_msg = message[:, :self.model_bits]
                    data_msg = message[:, self.model_bits:]
                    logger.info(f"Dual message extraction complete:")
                    logger.info(f"  Model bits: {model_msg}")
                    logger.info(f"  Data bits: {data_msg}")

            return message

        except Exception as e:
            logger.error(f"Error in detect: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detection failed: {str(e)}")
