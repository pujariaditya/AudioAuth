"""Generator model: embeds watermark bits into audio via SEANet encoder-decoder."""

import logging
import math
from typing import Dict, Optional, Tuple, List, Union, Any
import typing as tp

import numpy as np
import torch
import torch.nn as nn

from .modules.seanet import SEANetEncoder, SEANetDecoder
from ..config import GeneratorConfig

logger = logging.getLogger(__name__)

Array = Union[np.ndarray, List[float]]


class MsgProcessor(nn.Module):
    """Projects watermark bits into encoder's hidden space via learned embeddings.

    Each bit position has two learned embeddings (bit=0 and bit=1). The forward
    pass selects the appropriate embedding per bit and sums them into a single
    hidden-sized vector broadcast across time.

    Attributes:
        nbits: Number of watermark bits.
        hidden_size: Dimension of the encoder output (latent channels).
        msg_processor: Embedding table of shape (2*nbits, hidden_size).

    Args:
        nbits: Number of bits used to generate the message. Must be non-zero.
        hidden_size: Dimension of the encoder output.
    """

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """Select per-bit embeddings and add to encoder output.

        Args:
            hidden: Encoder output, shape (B, hidden, T).
            msg: Binary message, shape (B, nbits).

        Returns:
            Hidden state with watermark signal added, shape (B, hidden, T).
        """
        # Embedding layout: even indices = bit-0 embedding, odd = bit-1.
        # Adding msg (0 or 1) to even base indices selects the correct row.
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden

class Generator(nn.Module):
    """Audio generator with encoder-decoder architecture for watermark embedding.

    Embeds watermark bits into audio signals using a SEANet encoder-decoder.
    The encoder processes audio + watermark message into a latent representation;
    the decoder reconstructs the watermarked audio.

    Attributes:
        nbits: Number of watermark bits to embed.
        ratios: Downsampling ratios for each encoder/decoder layer.
        dimension: Latent representation channels.
        sample_rate: Audio sample rate in Hz.
        hop_length: Total downsampling factor (product of ratios).
        encoder: SEANet encoder for watermark embedding.
        decoder: SEANet decoder for audio reconstruction.
        message_embedding: Bottleneck message injection module.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """Initialize Generator with configuration.

        Args:
            config: GeneratorConfig containing all model parameters.
        """
        super().__init__()

        logger.info("Initializing Generator model")

        if config is None:
            error_msg = "GeneratorConfig is required - no fallback to defaults allowed"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(config, GeneratorConfig):
            error_msg = f"Expected GeneratorConfig, got {type(config).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        required_attrs = ['nbits', 'sample_rate', 'encoder_dim', 'encoder_rates',
                         'decoder_dim', 'decoder_rates', 'latent_dim']
        for attr in required_attrs:
            if not hasattr(config, attr):
                error_msg = f"GeneratorConfig missing required attribute: {attr}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.debug(f"Config validated: sample_rate={config.sample_rate}, nbits={config.nbits}")

        if config.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {config.sample_rate}. Must be positive.")

        try:
            sample_rate: int = config.sample_rate
            channels_audio: int = config.encoder_channels
            dimension: int = config.latent_dim
            nbits: int = config.nbits
            msg_dimension: int = nbits

            channels_enc: int = config.encoder_dim
            n_fft_base: int = config.encoder_n_fft_base
            n_residual_enc: int = config.encoder_residual_layers
            res_scale_enc: Optional[float] = config.encoder_res_scale
            strides: List[int] = config.encoder_rates

            if hasattr(config, "decoder_rates") and config.decoder_rates != strides:
                raise ValueError("decoder_rates must equal encoder_rates for symmetric up/down sampling.")
            activation: str = config.encoder_activation
            activation_kwargs: Dict[str, Any] = config.encoder_activation_params
            norm: str = config.encoder_norm
            norm_kwargs: Dict[str, Any] = config.encoder_norm_params
            kernel_size: int = config.encoder_kernel_size
            last_kernel_size: int = config.encoder_last_kernel_size
            residual_kernel_size: int = config.encoder_residual_kernel_size
            dilation_base: int = config.encoder_dilation_base
            skip: str = config.encoder_skip
            encoder_l2norm: bool = config.encoder_l2norm
            bias: bool = config.encoder_bias
            spec: str = config.encoder_spec
            spec_compression: str = config.encoder_spec_compression
            pad_mode: str = config.encoder_pad_mode
            causal: bool = config.encoder_causal
            zero_init: bool = config.encoder_zero_init
            inout_norm: bool = config.encoder_inout_norm
            act_all: bool = config.encoder_act_all
            expansion: int = config.encoder_expansion
            groups: int = config.encoder_groups
            encoder_wav_std: float = config.encoder_wav_std
            encoder_spec_means: List[float] = config.encoder_spec_means
            encoder_spec_stds: List[float] = config.encoder_spec_stds
            encoder_spec_learnable: bool = config.encoder_spec_learnable

            embedding_dim: int = config.encoder_embedding_dim
            embedding_layers: int = config.encoder_embedding_layers
            freq_bands: int = config.encoder_freq_bands
            freq_band_primary_weight: float = config.freq_band_primary_weight
            freq_band_secondary_weight: float = config.freq_band_secondary_weight
            freq_band_attenuation: float = config.freq_band_attenuation
            model_msg_bits: int = config.model_msg_bits
            data_msg_bits: int = config.data_msg_bits

            channels_dec: int = config.decoder_dim
            n_residual_dec: int = config.decoder_residual_layers
            res_scale_dec: Optional[float] = config.decoder_res_scale
            final_activation: str = config.decoder_final_activation
            final_activation_params: Dict[str, Any] = config.decoder_final_activation_params
            decoder_trim_right_ratio: float = config.decoder_trim_right_ratio
            decoder_wav_std: float = config.decoder_wav_std

            logger.debug(f"Parameters extracted: encoder_dim={channels_enc}, decoder_dim={channels_dec}, latent_dim={dimension}")

        except AttributeError as e:
            error_msg = f"Missing required config attribute: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

        self.nbits: int = nbits
        self.ratios: List[int] = strides
        self.dimension: int = dimension
        self.sample_rate: int = sample_rate
        self.hop_length: int = int(np.prod(self.ratios))

        logger.info(f"Generator initialized: {nbits} bits")
        logger.info(f"Core attributes set: nbits={self.nbits}, hop_length={self.hop_length}, sample_rate={self.sample_rate}")

        try:
            logger.debug(f"Initializing SEANetEncoder with msg_dimension={msg_dimension}")
            self.encoder = SEANetEncoder(
                channels=channels_audio,
                dimension=dimension,
                msg_dimension=msg_dimension,
                n_filters=channels_enc,
                n_fft_base=n_fft_base,
                n_residual_layers=n_residual_enc,
                ratios=strides,
                activation=activation,
                activation_params=activation_kwargs,
                norm=norm,
                norm_params=norm_kwargs,
                kernel_size=kernel_size,
                last_kernel_size=last_kernel_size,
                residual_kernel_size=residual_kernel_size,
                dilation_base=dilation_base,
                skip=skip,
                causal=causal,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                l2norm=encoder_l2norm,
                bias=bias,
                spec=spec,
                spec_compression=spec_compression,
                spec_learnable=encoder_spec_learnable,
                res_scale=res_scale_enc,
                wav_std=encoder_wav_std,
                spec_means=encoder_spec_means,
                spec_stds=encoder_spec_stds,
                pad_mode=pad_mode,
                zero_init=zero_init,
                inout_norm=inout_norm,
                embedding_dim=embedding_dim,
                embedding_layers=embedding_layers,
                freq_bands=freq_bands,
                freq_band_primary_weight=freq_band_primary_weight,
                freq_band_secondary_weight=freq_band_secondary_weight,
                freq_band_attenuation=freq_band_attenuation,
                model_msg_bits=model_msg_bits,
                data_msg_bits=data_msg_bits,
                film_epsilon=config.film_epsilon,
                film_residual_alpha=config.film_residual_alpha,
                film_start_block=config.film_start_block,
                embedding_gate_layers=config.embedding_gate_layers,
                enable_layer_norm_embedding=config.enable_layer_norm_embedding
            )
            logger.info("SEANetEncoder initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize SEANetEncoder: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        try:
            logger.debug("Initializing SEANetDecoder")
            self.decoder = SEANetDecoder(
                channels=channels_audio,
                dimension=dimension,
                n_filters=channels_dec,
                n_residual_layers=n_residual_dec,
                ratios=strides,
                activation=activation,
                activation_params=activation_kwargs,
                norm=norm,
                norm_params=norm_kwargs,
                kernel_size=kernel_size,
                last_kernel_size=last_kernel_size,
                residual_kernel_size=residual_kernel_size,
                dilation_base=dilation_base,
                skip=skip,
                causal=causal,
                trim_right_ratio=decoder_trim_right_ratio,
                final_activation=final_activation,
                final_activation_params=final_activation_params,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                bias=bias,
                res_scale=res_scale_dec,
                wav_std=decoder_wav_std,
                pad_mode=pad_mode,
                zero_init=zero_init,
                inout_norm=inout_norm
            )
            logger.info("SEANetDecoder initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize SEANetDecoder: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        logger.info("Generator initialization complete")

        self.message_embedding = MsgProcessor(nbits=self.nbits, hidden_size=self.dimension)

    def preprocess(self, audio_data: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """Pad audio to the nearest multiple of hop_length.

        Args:
            audio_data: Input audio tensor of shape (B, C, T).
            sample_rate: Audio sample rate in Hz. If None, uses self.sample_rate.

        Returns:
            Padded audio tensor.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        if sample_rate != self.sample_rate:
            error_msg = f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            length = audio_data.shape[-1]
            right_pad = math.ceil(length / self.hop_length) * self.hop_length - length

            if right_pad > 0:
                logger.debug(f"Applying right padding of {right_pad} samples")
                audio_data = nn.functional.pad(audio_data, (0, right_pad))

            return audio_data

        except Exception as e:
            error_msg = f"Error in preprocessing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def encode(
        self,
        audio_data: torch.Tensor,
        msg: torch.Tensor
    ) -> torch.Tensor:
        """Encode audio with watermark message into latent representation.

        Args:
            audio_data: Audio tensor of shape (B, 1, T).
            msg: Watermark message tensor of shape (B, nbits), binary values.

        Returns:
            Latent tensor of shape (B, D, T') where T' = T / hop_length.
        """
        try:
            if audio_data.dim() != 3 or audio_data.size(1) != 1:
                error_msg = f"Expected audio_data shape [B, 1, T], got {list(audio_data.shape)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if msg.dim() != 2 or msg.size(1) != self.nbits:
                error_msg = f"Expected msg shape [B, {self.nbits}], got {list(msg.shape)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if torch.isnan(audio_data).any() or torch.isinf(audio_data).any():
                logger.warning("NaN or Inf detected in audio_data")

            if torch.isnan(msg).any() or torch.isinf(msg).any():
                logger.warning("NaN or Inf detected in msg")

            logger.debug(f"Encoding audio shape {list(audio_data.shape)} with message shape {list(msg.shape)}")

            z = self.encoder(audio_data, msg)

            logger.debug(f"Encoded to latent shape {list(z.shape)}")
            return z

        except Exception as e:
            error_msg = f"Encoding failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to audio signal.

        Args:
            z: Latent tensor of shape (B, D, T').

        Returns:
            Reconstructed audio tensor of shape (B, 1, T).
        """
        try:
            if z.dim() != 3:
                error_msg = f"Expected z shape [B, D, T'], got {list(z.shape)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if z.size(1) != self.dimension:
                error_msg = f"Expected latent dimension {self.dimension}, got {z.size(1)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if torch.isnan(z).any() or torch.isinf(z).any():
                logger.warning("NaN or Inf detected in latent representation")

            logger.debug(f"Decoding latent shape {list(z.shape)}")

            x = self.decoder(z)

            logger.debug(f"Decoded to audio shape {list(x.shape)}")
            return x

        except Exception as e:
            error_msg = f"Decoding failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def forward(
        self,
        audio_signal: Union[torch.Tensor, Any],
        msg: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """Encode audio with watermark, inject message at bottleneck, and decode.

        Args:
            audio_signal: Audio tensor (B, 1, T) or object with 'audio_data' attr.
            msg: Watermark message tensor of shape (B, nbits), binary values.
            sample_rate: Audio sample rate in Hz. If None, uses self.sample_rate.

        Returns:
            Watermarked audio tensor of shape (B, 1, T), same length as input.
        """
        try:
            if hasattr(audio_signal, 'audio_data'):
                length = audio_signal.audio_data.shape[-1]
                audio_data = audio_signal.audio_data
                logger.debug("Processing AudioSignal-like input")
            else:
                length = audio_signal.shape[-1]
                audio_data = audio_signal
                logger.debug("Processing direct tensor input")

            if sample_rate is not None and sample_rate != self.sample_rate:
                error_msg = f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            audio_data = self.preprocess(audio_data, sample_rate)
            z = self.encode(audio_data, msg)
            z = self.message_embedding(z, msg)
            x = self.decode(z)

            result = x[..., :length]

            return result

        except Exception as e:
            error_msg = f"Forward pass failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
