"""Configuration system for AudioAuth.

Pydantic-based configuration system for watermarking framework training with GAN losses.
Implements the AudioAuth architecture.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal, Any, Dict, Union
import yaml

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.v1.utils import deep_update
from pydantic_settings import BaseSettings, CliSettingsSource, YamlConfigSettingsSource


# --- Generator & Detector Architecture ---

class GeneratorConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Generator configuration with encoder and decoder parameters.

    Groups core model dimensions, audio settings, encoder/decoder layer
    details, and watermark-specific FiLM modulation parameters.
    """

    nbits: int = 32
    model_msg_bits: int = 16
    data_msg_bits: int = 16
    encoder_dim: int = 64
    encoder_rates: List[int] = [8, 5, 4, 2]
    encoder_residual_layers: int = 2
    decoder_dim: int = 96
    decoder_rates: List[int] = [8, 5, 4, 2]
    decoder_residual_layers: int = 3
    latent_dim: int = 128

    sample_rate: int = 16000
    channels: int = 1

    # Encoder detailed parameters
    encoder_channels: int = 1
    encoder_n_fft_base: int = 64
    encoder_activation: str = 'ELU'
    encoder_activation_params: Dict[str, Any] = Field(default_factory=lambda: {'alpha': 1.0})
    encoder_norm: str = 'weight_norm'
    encoder_norm_params: Dict[str, Any] = Field(default_factory=dict)
    encoder_kernel_size: int = 5
    encoder_last_kernel_size: int = 5
    encoder_residual_kernel_size: int = 5
    encoder_dilation_base: int = 1
    encoder_skip: str = 'identity'
    encoder_pad_mode: str = 'constant'
    encoder_causal: bool = True
    encoder_act_all: bool = False
    encoder_expansion: int = 1
    encoder_groups: int = -1
    encoder_l2norm: bool = True
    encoder_bias: bool = True
    encoder_spec: str = 'stft'
    encoder_spec_compression: str = 'log'
    encoder_spec_learnable: bool = False
    encoder_res_scale: Optional[float] = 0.5773502691896258
    encoder_wav_std: float = 0.1122080159
    encoder_spec_means: List[float] = Field(default_factory=lambda: [-4.554, -4.315, -4.021, -3.726, -3.477])
    encoder_spec_stds: List[float] = Field(default_factory=lambda: [2.830, 2.837, 2.817, 2.796, 2.871])
    encoder_zero_init: bool = True
    encoder_inout_norm: bool = True

    # Watermarking parameters
    encoder_embedding_dim: int = 64
    encoder_embedding_layers: int = 2
    encoder_freq_bands: int = 4

    # Frequency band configuration for dual watermarks
    freq_bands: int = 4
    freq_band_primary_weight: float = 0.7
    freq_band_secondary_weight: float = 0.3
    freq_band_attenuation: float = 0.3

    # FiLM modulation parameters for gentle watermarking
    film_epsilon: float = 0.1  # Lower = gentler modulation
    film_residual_alpha: float = 0.3
    film_start_block: int = 1  # 0 = first block
    embedding_gate_layers: int = 2
    enable_layer_norm_embedding: bool = True

    # Decoder detailed parameters
    decoder_channels: int = 1
    decoder_activation: str = 'ELU'
    decoder_activation_params: Dict[str, Any] = Field(default_factory=lambda: {'alpha': 1.0})
    decoder_norm: str = 'weight_norm'
    decoder_norm_params: Dict[str, Any] = Field(default_factory=dict)
    decoder_kernel_size: int = 5
    decoder_last_kernel_size: int = 5
    decoder_residual_kernel_size: int = 5
    decoder_dilation_base: int = 1
    decoder_skip: str = 'identity'
    decoder_pad_mode: str = 'constant'
    decoder_causal: bool = True
    decoder_trim_right_ratio: float = 1.0
    decoder_final_activation: str = 'Tanh'
    decoder_final_activation_params: Dict[str, Any] = Field(default_factory=dict)
    decoder_act_all: bool = False
    decoder_expansion: int = 1
    decoder_groups: int = -1
    decoder_bias: bool = True
    decoder_res_scale: Optional[float] = 0.5773502691896258
    decoder_wav_std: float = 0.1122080159
    decoder_zero_init: bool = True
    decoder_inout_norm: bool = True

    @property
    def downsample_rate(self) -> int:
        """Total downsampling rate."""
        rate = 1
        for s in self.encoder_rates:
            rate *= s
        return rate

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz."""
        return self.sample_rate / self.downsample_rate


class DetectorConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Detector configuration for watermark message extraction.

    Groups core model dimensions, detection thresholds, dual watermark
    bit allocation, and encoder layer parameters.
    """

    encoder_dim: int = 64
    encoder_rates: List[int] = [8, 5, 4, 2]
    encoder_residual_layers: int = 2
    latent_dim: int = 128
    sample_rate: int = 16000

    # Detection-specific parameters
    nbits: int = 32
    output_dim: int = 32
    detection_threshold: float = 0.5

    # Dual watermark extraction parameters
    model_bits: int = 16
    data_bits: int = 16

    # Encoder detailed parameters
    encoder_channels: int = 1
    encoder_n_fft_base: int = 64
    encoder_activation: str = 'ELU'
    encoder_activation_params: Dict[str, Any] = Field(default_factory=lambda: {'alpha': 1.0})
    encoder_norm: str = 'weight_norm'
    encoder_norm_params: Dict[str, Any] = Field(default_factory=dict)
    encoder_kernel_size: int = 5
    encoder_last_kernel_size: int = 5
    encoder_residual_kernel_size: int = 5
    encoder_dilation_base: int = 1
    encoder_skip: str = 'identity'
    encoder_pad_mode: str = 'constant'
    encoder_causal: bool = True
    encoder_act_all: bool = False
    encoder_expansion: int = 1
    encoder_groups: int = -1
    encoder_l2norm: bool = True
    encoder_bias: bool = True
    encoder_spec: str = 'stft'
    encoder_spec_compression: str = 'log'
    encoder_spec_learnable: bool = False
    encoder_res_scale: Optional[float] = 0.5773502691896258
    encoder_wav_std: float = 0.1122080159
    encoder_spec_means: List[float] = Field(default_factory=lambda: [-4.554, -4.315, -4.021, -3.726, -3.477])
    encoder_spec_stds: List[float] = Field(default_factory=lambda: [2.830, 2.837, 2.817, 2.796, 2.871])
    encoder_zero_init: bool = True
    encoder_inout_norm: bool = True

    @property
    def downsample_rate(self) -> int:
        """Total downsampling rate."""
        rate = 1
        for s in self.encoder_rates:
            rate *= s
        return rate

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz."""
        return self.sample_rate / self.downsample_rate


class LocatorConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Locator configuration for precise watermark temporal localization.

    Groups core model dimensions, localization thresholds, and encoder
    layer parameters. Outputs a single channel of logits indicating
    watermark presence at each time step.
    """

    encoder_dim: int = 64
    encoder_rates: List[int] = [8, 5, 4, 2]
    encoder_residual_layers: int = 2
    latent_dim: int = 128
    sample_rate: int = 16000

    output_dim: int = 32
    localization_threshold: float = 0.5

    # Encoder detailed parameters
    encoder_channels: int = 1
    encoder_n_fft_base: int = 64
    encoder_activation: str = 'ELU'
    encoder_activation_params: Dict[str, Any] = Field(default_factory=lambda: {'alpha': 1.0})
    encoder_norm: str = 'weight_norm'
    encoder_norm_params: Dict[str, Any] = Field(default_factory=dict)
    encoder_kernel_size: int = 5
    encoder_last_kernel_size: int = 5
    encoder_residual_kernel_size: int = 5
    encoder_dilation_base: int = 1
    encoder_skip: str = 'identity'
    encoder_pad_mode: str = 'constant'
    encoder_causal: bool = True
    encoder_act_all: bool = False
    encoder_expansion: int = 1
    encoder_groups: int = -1
    encoder_l2norm: bool = True
    encoder_bias: bool = True
    encoder_spec: str = 'stft'
    encoder_spec_compression: str = 'log'
    encoder_spec_learnable: bool = False
    encoder_res_scale: Optional[float] = 0.5773502691896258
    encoder_wav_std: float = 0.1122080159
    encoder_spec_means: List[float] = Field(default_factory=lambda: [-4.554, -4.315, -4.021, -3.726, -3.477])
    encoder_spec_stds: List[float] = Field(default_factory=lambda: [2.830, 2.837, 2.817, 2.796, 2.871])
    encoder_zero_init: bool = True
    encoder_inout_norm: bool = True

    # Single logit channel for watermark presence
    output_channels: int = 1

    @field_validator('output_channels', mode='after')
    @classmethod
    def validate_output_channels(cls, v):
        """Ensure output_channels is 1 for single-channel watermark detection."""
        if v != 1:
            raise ValueError(
                f"output_channels must be 1 for single-channel watermark detection "
                f"(logits for watermark presence). Got {v}."
            )
        return v

    @field_validator('encoder_spec_means', mode='after')
    @classmethod
    def validate_spec_means_size(cls, v, info):
        """Ensure spec_means has N+1 values for N encoder rates (SEANetEncoder requirement)."""
        if 'encoder_rates' in info.data:
            expected_len = len(info.data['encoder_rates']) + 1
            if len(v) != expected_len:
                raise ValueError(
                    f"encoder_spec_means must have {expected_len} values "
                    f"(N+1 for N encoder rates). Got {len(v)} values for "
                    f"{len(info.data['encoder_rates'])} encoder rates."
                )
        return v

    @field_validator('encoder_spec_stds', mode='after')
    @classmethod
    def validate_spec_stds_size(cls, v, info):
        """Ensure spec_stds has N+1 values for N encoder rates (SEANetEncoder requirement)."""
        if 'encoder_rates' in info.data:
            expected_len = len(info.data['encoder_rates']) + 1
            if len(v) != expected_len:
                raise ValueError(
                    f"encoder_spec_stds must have {expected_len} values "
                    f"(N+1 for N encoder rates). Got {len(v)} values for "
                    f"{len(info.data['encoder_rates'])} encoder rates."
                )
        return v

    @property
    def downsample_rate(self) -> int:
        """Total downsampling rate."""
        rate = 1
        for s in self.encoder_rates:
            rate *= s
        return rate

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz."""
        return self.sample_rate / self.downsample_rate


# --- Attack Configurations ---

class LocalizationAttacksConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Localization attacks configuration for segment-level modifications.

    Controls how audio segments are selectively replaced to create
    ground truth masks for localization training.
    """

    window_duration: float = Field(..., gt=0, description="Duration of segments for localization (in seconds) (required)")
    target_ratio: float = Field(..., ge=0, le=1, description="Target ratio of segments to attack (0-1) (required)")
    original_revert_prob: float = Field(..., ge=0, le=1, description="Probability of reverting to original (required)")
    zero_replace_prob: float = Field(..., ge=0, le=1, description="Probability of replacing with zeros (required)")

    @field_validator('original_revert_prob', 'zero_replace_prob', mode='after')
    @classmethod
    def validate_probabilities_sum(cls, v, info):
        """Ensure probabilities approximately sum to 1."""
        if all(k in info.data for k in ['original_revert_prob', 'zero_replace_prob']):
            total = info.data['original_revert_prob'] + info.data['zero_replace_prob']
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"Localization probabilities must sum to ~1.0, got {total:.3f} "
                    f"(original_revert: {info.data['original_revert_prob']:.3f}, "
                    f"zero_replace: {info.data['zero_replace_prob']:.3f})"
                )
        return v


class SequenceAttacksConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Sequence attacks configuration for batch-level transformations.

    Defines probabilities for structural audio manipulations (reversing,
    trimming, cropping, shuffling) that test temporal robustness.
    """

    reverse_prob: float = Field(..., ge=0, le=1, description="Probability of reversing audio (required)")
    head_trim_prob: float = Field(..., ge=0, le=1, description="Probability of trimming samples from head (required)")
    tail_trim_prob: float = Field(..., ge=0, le=1, description="Probability of trimming samples from tail (required)")
    crop_replacement_prob: float = Field(..., ge=0, le=1, description="Probability of crop replacement (required)")
    shuffle_prob: float = Field(..., ge=0, le=1, description="Probability of shuffling segments (required)")
    chunk_shuffle_prob: float = Field(..., ge=0, le=1, description="Probability of chunk shuffle (required)")
    segment_duration: float = Field(..., gt=0, description="Duration for shuffle segments (in seconds) (required)")
    chunk_divisions: int = Field(..., ge=2, description="Number of chunks for chunk shuffle (required)")
    max_trim_ms: float = Field(12.0, gt=0, le=25.0, description="Maximum trim duration in milliseconds (default: 12ms, addresses ~100 sample issue)")

    @field_validator('reverse_prob', 'head_trim_prob', 'tail_trim_prob', 'crop_replacement_prob', 'shuffle_prob', 'chunk_shuffle_prob', mode='after')
    @classmethod
    def validate_probabilities_sum(cls, v, info):
        """Ensure probabilities approximately sum to 1."""
        prob_fields = ['reverse_prob', 'head_trim_prob', 'tail_trim_prob', 'crop_replacement_prob', 'shuffle_prob', 'chunk_shuffle_prob']
        if all(k in info.data for k in prob_fields):
            total = sum(info.data[k] for k in prob_fields)
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"Sequence probabilities must sum to ~1.0, got {total:.3f} "
                    f"(reverse: {info.data['reverse_prob']:.3f}, "
                    f"head_trim: {info.data['head_trim_prob']:.3f}, "
                    f"tail_trim: {info.data['tail_trim_prob']:.3f}, "
                    f"crop_replacement: {info.data['crop_replacement_prob']:.3f}, "
                    f"shuffle: {info.data['shuffle_prob']:.3f}, "
                    f"chunk_shuffle: {info.data['chunk_shuffle_prob']:.3f})"
                )
        return v


# --- Effect Parameter Types ---

class WhiteNoiseParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for white noise effect."""
    snr_db: float = Field(20.0, ge=0, le=50, description="Signal-to-noise ratio in dB")


class PinkNoiseParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for pink noise effect."""
    noise_std: float = Field(0.01, ge=0.0, le=1.0, description="Standard deviation of pink noise")


class LowpassParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for lowpass filter effect."""
    cutoff_freq: float = Field(4000.0, ge=100, le=20000, description="Cutoff frequency in Hz")


class HighpassParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for highpass filter effect."""
    cutoff_freq: float = Field(1000.0, ge=20, le=10000, description="Cutoff frequency in Hz")


class VolumeParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for volume change effect."""
    volume_factor: float = Field(0.8, ge=0.1, le=2.0, description="Volume multiplication factor")


class UpdownResampleParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for updown resample effect."""
    intermediate_freq: int = Field(32000, ge=8000, le=96000, description="Intermediate frequency for resampling in Hz")


class EchoParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for echo/reverb effect."""
    volume_range: List[float] = Field([0.1, 0.5], description="Volume range for echo (min, max)")
    duration_range: List[float] = Field([0.1, 0.5], description="Duration range for echo in seconds (min, max)")


class Mp3CompressionParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for MP3 compression effect."""
    bitrate: str = Field("128k", description="MP3 bitrate (e.g., '64k', '128k', '192k', '320k')")


class BoostAudioParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for boost audio effect."""
    amount: float = Field(20.0, ge=0.0, le=100.0, description="Boost amount in percentage")


class DuckAudioParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for duck audio effect."""
    amount: float = Field(20.0, ge=0.0, le=100.0, description="Duck amount in percentage")


class SpeedParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for speed change effect."""
    speed_range: List[float] = Field([0.8, 1.2], description="Speed factor range (min, max)")

class RandomNoiseParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for random Gaussian noise effect."""
    noise_std: float = Field(0.001, ge=0.0, le=1.0, description="Standard deviation of Gaussian noise")


class SmoothParams(BaseModel, extra="forbid", validate_assignment=True):
    """Parameters for smoothing filter effect."""
    window_size_range: List[float] = Field([2.0, 10.0], description="Window size range for smoothing (min, max)")


class EffectParamsConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Configuration for individual effect parameters.

    Each effect uses choices format for the scheduler, e.g.
    ``pink_noise: {noise_std: {choices: [0.005, 0.01, 0.02]}}``.
    Bandpass uses ``frequency_pairs`` format:
    ``bandpass: {frequency_pairs: {choices: [[200, 3000], ...]}}``.
    """
    white_noise: Optional[Dict[str, Any]] = Field(default_factory=dict)
    pink_noise: Optional[Dict[str, Any]] = Field(default_factory=dict)
    lowpass: Optional[Dict[str, Any]] = Field(default_factory=dict)
    highpass: Optional[Dict[str, Any]] = Field(default_factory=dict)
    bandpass: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Must use frequency_pairs format: {frequency_pairs: {choices: [[low, high], ...]}}"
    )
    volume: Optional[Dict[str, Any]] = Field(default_factory=dict)
    updown_resample: Optional[Dict[str, Any]] = Field(default_factory=dict)
    echo: Optional[Dict[str, Any]] = Field(default_factory=dict)
    boost_audio: Optional[Dict[str, Any]] = Field(default_factory=dict)
    duck_audio: Optional[Dict[str, Any]] = Field(default_factory=dict)
    speed: Optional[Dict[str, Any]] = Field(default_factory=dict)
    random_noise: Optional[Dict[str, Any]] = Field(default_factory=dict)
    smooth: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mp3_compression: Optional[Dict[str, Any]] = Field(default_factory=dict, description="MP3 codec compression effect")
    aac_compression: Optional[Dict[str, Any]] = Field(default_factory=dict, description="AAC codec compression effect")
    encodec_compression: Optional[Dict[str, Any]] = Field(default_factory=dict, description="EnCodec neural codec compression effect")
    identity: Optional[Dict[str, Any]] = Field(default={}, description="Identity effect has no parameters")


# --- Scheduler & Composite Attack Config ---

class SchedulerConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Configuration for EffectScheduler adaptive selection system.

    Controls the EMA-based difficulty tracking and softmax temperature
    annealing that bias training toward harder effects over time.
    """
    beta: float = Field(0.9, ge=0.0, lt=1.0, description="EMA smoothing factor for metrics (higher = more weight on history)")
    ber_threshold: float = Field(0.001, ge=0.0, le=1.0, description="Success threshold for BER (lower is better)")
    miou_threshold: float = Field(0.95, ge=0.0, le=1.0, description="Success threshold for mIoU (higher is better)")
    # Temperature annealing: softmax(difficulty / tau) controls how
    # aggressively the scheduler favors harder effects.
    temperature_start: float = Field(1.0, gt=0.0, description="Initial softmax temperature for effect selection")
    temperature_end: float = Field(0.7, gt=0.0, description="Final softmax temperature after annealing")


class EffectAttacksConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Effect attacks configuration for audio effect transformations with adaptive scheduler.

    Groups the per-effect enable flags, scheduler settings for adaptive
    difficulty, per-effect parameter choices, and compound chain options.
    """

    effect_enabled: Dict[str, bool] = Field(
        ...,
        description="Enable/disable individual effects (required - no defaults)"
    )

    scheduler_config: SchedulerConfig = Field(
        ...,
        description="Configuration for EffectScheduler adaptive selection system (uses dual metrics: BER and mIoU)"
    )

    effect_params: EffectParamsConfig = Field(
        default_factory=EffectParamsConfig,
        description="Parameters for individual effects with choices format"
    )

    # Compound distortion chains: probability of applying multiple
    # effects sequentially within a single training step
    compound_chain_prob: float = Field(0.0, ge=0.0, le=1.0, description="Probability of applying compound distortion chains per batch")
    max_chain_length: int = Field(3, ge=2, le=5, description="Maximum number of effects in a compound chain")

    @field_validator('effect_enabled', mode='after')
    @classmethod
    def validate_at_least_one_enabled(cls, v):
        """Ensure at least one effect is enabled."""
        if v and not any(v.values()):
            raise ValueError("At least one effect must be enabled")
        return v


class AttacksConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Complete attacks configuration for watermarking robustness.

    Combines localization (segment-level), sequence (structural), and
    effect (signal processing) attack sub-configs.
    """

    enabled: bool = Field(..., description="Whether attacks are enabled (required)")
    localization: LocalizationAttacksConfig = Field(..., description="Localization attacks settings (required)")
    sequence: SequenceAttacksConfig = Field(..., description="Sequence attacks settings (required)")
    effect: EffectAttacksConfig = Field(..., description="Effect attacks settings (required)")


# --- Watermarking System Config ---

class WatermarkingConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Complete watermarking system configuration.

    Ties together generator/detector/locator architecture configs,
    dual watermark bit allocation, phase identifiers, and per-phase
    attack configurations.
    """

    nbits: int = Field(..., gt=0, description="Number of watermark bits (must be positive)")
    sample_rate: int = Field(..., gt=0, description="System-wide sample rate in Hz (must match generator/detector)")

    model_bits: int = Field(16, description="Number of model watermark bits")
    data_bits: int = Field(16, description="Number of data watermark bits")
    model_pattern: List[int] = Field(..., description="Fixed 16-bit pattern for model identification")

    train_phase: str = Field(..., min_length=1, description="Training phase identifier (required)")
    valid_phase: str = Field(..., min_length=1, description="Validation phase identifier (required)")
    audio_sample_phase: str = Field(..., min_length=1, description="Audio sample generation phase identifier (required)")

    generator: GeneratorConfig = Field(..., description="Generator configuration (required)")
    detector: DetectorConfig = Field(..., description="Detector configuration (required)")
    locator: LocatorConfig = Field(..., description="Locator configuration (required)")
    train_attacks: AttacksConfig = Field(..., description="Training-specific attacks configuration (required)")
    valid_attacks: AttacksConfig = Field(..., description="Validation-specific attacks configuration (required)")

    @field_validator('generator', 'detector', 'locator', mode='after')
    @classmethod
    def validate_sample_rates(cls, v, info):
        """Ensure sample rates are consistent across all components."""
        if 'sample_rate' in info.data:
            system_sample_rate = info.data['sample_rate']
            if hasattr(v, 'sample_rate') and v.sample_rate != system_sample_rate:
                component_name = info.field_name
                raise ValueError(
                    f"{component_name.capitalize()} sample_rate ({v.sample_rate}Hz) must match "
                    f"system sample_rate ({system_sample_rate}Hz). "
                    f"Please ensure all sample_rate values are consistent."
                )
        return v

    @field_validator('nbits', mode='after')
    @classmethod
    def validate_nbits_consistency(cls, v, info):
        """Ensure nbits is always 32 and matches across all components."""
        if v != 32:
            raise ValueError(f"nbits must be 32 for dual watermarking, got {v}")

        if 'generator' in info.data and hasattr(info.data['generator'], 'nbits'):
            if info.data['generator'].nbits != v:
                raise ValueError(
                    f"Generator nbits ({info.data['generator'].nbits}) must be 32"
                )
        if 'detector' in info.data and hasattr(info.data['detector'], 'nbits'):
            if info.data['detector'].nbits != v:
                raise ValueError(
                    f"Detector nbits ({info.data['detector'].nbits}) must be 32"
                )

        if 'model_bits' in info.data and 'data_bits' in info.data:
            if info.data['model_bits'] + info.data['data_bits'] != 32:
                raise ValueError(
                    f"model_bits ({info.data['model_bits']}) + data_bits ({info.data['data_bits']}) "
                    f"must equal 32"
                )

        return v

    @field_validator('model_pattern', mode='after')
    @classmethod
    def validate_model_pattern(cls, v, info):
        """Ensure model_pattern is exactly 16 bits with values 0 or 1."""
        if len(v) != 16:
            raise ValueError(f"model_pattern must be exactly 16 bits, got {len(v)}")

        for i, bit in enumerate(v):
            if bit not in [0, 1]:
                raise ValueError(f"model_pattern[{i}] must be 0 or 1, got {bit}")

        return v

    @field_validator('train_phase', 'valid_phase', 'audio_sample_phase', mode='after')
    @classmethod
    def validate_phase_uniqueness(cls, v, info):
        """Ensure phase identifiers are unique."""
        phases = []
        for phase_name in ['train_phase', 'valid_phase', 'audio_sample_phase']:
            if phase_name in info.data:
                phases.append(info.data[phase_name])

        if len(phases) == len(set(phases)):
            return v
        else:
            raise ValueError(
                f"Phase identifiers must be unique. Got train_phase='{info.data.get('train_phase')}', "
                f"valid_phase='{info.data.get('valid_phase')}', "
                f"audio_sample_phase='{info.data.get('audio_sample_phase')}'"
            )


# --- GAN Discriminator ---

class DiscriminatorConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Discriminator configuration for GAN training.

    Groups discriminator type selection, multi-period (MPD),
    multi-scale (MSD), and multi-resolution (MRD) settings.
    """

    use_discriminator: bool = True
    discriminator_type: Literal["mpd_msd_mrd", "mpd_msd", "mpd"] = "mpd_msd_mrd"
    period_scales: List[int] = [2, 3, 5, 7, 11]
    scale_downsample_rates: List[int] = []  # Empty = disable MSD
    # Use different factors like [1, 2, 4] for progressive downsampling;
    # identical values like [4, 4, 4, 4] create redundant discriminators.
    fft_sizes: List[int] = [2048, 1024, 512]
    bands: List[List[float]] = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    sample_rate: int = 16000


# --- Optimizer & Scheduler ---

class OptimizerConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Optimizer configuration with unified optimizer for watermarking system.

    Groups watermarking optimizer settings (shared across generator,
    detector, locator), discriminator optimizer settings (kept separate
    because adversarial training uses different dynamics), LR scheduler,
    and gradient clipping.
    """

    optimizer: Literal["adamw", "adamp"] = "adamw"
    learning_rate: float = 1e-4
    betas: List[float] = [0.8, 0.99]
    eps: float = 1e-8
    weight_decay: float = 0.0

    d_optimizer: Literal["adamw", "adamp"] = "adamw"
    d_learning_rate: float = 1e-4
    d_betas: List[float] = [0.8, 0.99]
    d_eps: float = 1e-8
    d_weight_decay: float = 0.0

    # LR scheduler
    scheduler_type: Literal["constant", "exponential", "cosine", "linear_warmup_cosine", "cosine_cyclic"] = "constant"
    warmup_steps: int = 0
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    gamma: float = 0.999996

    # Cosine cyclic scheduler parameters
    cycle_length: int = 5000
    min_lr_factor: float = 0.5
    num_cycles: Optional[int] = None
    max_lr: float = 1.0e-3

    # Gradient clipping
    watermarking_grad_clip_norm: float = 1000.0
    discriminator_grad_clip_norm: float = 10.0


# --- Loss ---

class LossConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Loss function configuration aligned with AudioAuth.

    Groups reconstruction losses (mel, waveform, multi-scale STFT),
    watermark-specific losses (decoding, localization), perceptual
    losses (TF loudness), and GAN adversarial losses.
    """

    mel_loss_weight: float = 0.0  # Disabled in favor of multi-scale STFT
    mel_loss_params: Dict[str, Any] = {
        "sample_rate": 16000,
        "n_fft": 2048,
        "win_lengths": [32, 64, 128, 256, 512, 1024, 2048],
        "n_mels": [5, 10, 20, 40, 80, 160, 320],
        "pow": 1.0,
        "normalized": False,
        "eps": 1e-5,
        "mag_weight": 0.0
    }

    waveform_loss_weight: float = 1000.0  # L1 sample-level reconstruction

    stft_loss_weight: float = 10.0
    stft_loss_params: Dict[str, Any] = {
        "window_lengths": [2048, 512],
        "overlap": 0.75,
        "eps": 1e-7,
        "pow": 2.0,
        "mag_weight": 1.0,
        "log_weight": 1.0
    }

    # Decoding loss: extracts watermark bits from detector outputs
    # where mask=1
    decoding_loss_weight: float = 10000
    decoding_loss_params: Dict[str, Any] = {
        "pos_weight": 1.0,
        "neg_weight": 1.0
    }

    # Localization loss: BCE between locator output and ground truth mask
    localization_loss_weight: float = 100
    localization_loss_params: Dict[str, Any] = {
        "pos_weight": 1.0,
        "neg_weight": 1.0
    }

    # TF loudness loss: perceptual loudness difference across frequency bands
    tf_loudness_loss_weight: float = 1000.0
    tf_loudness_loss_params: Dict[str, Any] = {
        "num_freq_bands": 10,
        "window_size": 6400,
        "overlap": 1600,
        "sample_rate": 16000
    }

    # GAN losses
    use_gan_loss: bool = True
    adv_gen_loss_weight: float = 40.0
    adv_feat_loss_weight: float = 40.0

    discriminator_update_freq: int = 1


# --- Metrics ---

class MetricsConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Evaluation metrics configuration.

    Groups audio quality metrics (SI-SNR, PESQ, STOI), watermark
    accuracy thresholds (IoU, BER), and the list of metrics to log
    during validation.
    """

    use_sisnr: bool = True
    sisnr_epsilon: float = 1e-8
    sisnr_scaling: bool = True

    use_pesq: bool = True
    pesq_sample_rate: int = 16000  # PESQ only supports 8kHz or 16kHz
    pesq_mode: Literal["wb", "nb"] = "wb"
    disable_pesq_resampling: bool = False

    use_stoi: bool = True
    stoi_extended: bool = False

    iou_threshold: float = 0.5

    validation_metrics: List[str] = ["sisnr", "pesq", "stoi", "model_loc_acc", "data_loc_acc", "model_loc_fpr", "data_loc_fpr", "model_miou", "data_miou", "model_ber", "data_ber", "model_localized_ber", "data_localized_ber"]


# --- Dataset ---

class DatasetConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Dataset configuration.

    Groups manifest paths, segment/duration settings, data loader
    tuning, and validation cropping strategy.
    """
    train_manifest: Path = Path("data/train.jsonl")
    valid_manifest: Path = Path("data/valid.jsonl")
    test_manifest: Path = Path("data/test.jsonl")

    segment_length: int = 16000
    hop_length: Optional[int] = None
    audio_backend: Literal["torchaudio", "soundfile"] = "torchaudio"

    # Duration overrides (seconds) -- if set, override segment_length
    train_duration: Optional[float] = None
    valid_duration: Optional[float] = None
    test_duration: Optional[float] = None

    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    valid_cropping: Literal["random", "start", "center"] = "start"
    test_cropping: Literal["random", "start", "center"] = "start"

    @field_validator("train_manifest", "valid_manifest", "test_manifest", mode="after")
    @classmethod
    def check_files(cls, path: Path) -> Path:
        """Validate manifest files exist."""
        if not path.exists():
            raise ValueError(f"Manifest file {path} does not exist")
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Manifest file {path} must be a JSONL file")
        return path


# --- Training Run ---

class RunConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Training run configuration.

    Groups basic run identity, training loop settings, validation and
    checkpointing intervals, logging backends, hardware/distributed
    settings, cloud storage, debugging, and early stopping.
    """

    seed: int = 42
    output_dir: Path = Path("outputs/")
    experiment_name: Optional[str] = None
    job_id: Optional[str] = None

    max_iterations: int = 100000
    batch_size_train: int = 4
    batch_size_eval: int = 8
    accum_grad_iters: int = 4

    validation_interval: int = 1000
    num_valid_samples: Optional[int] = None
    num_valid_batches: int = 10
    save_audio_samples: bool = True
    num_audio_samples: int = 4
    sample_freq: Optional[int] = None  # If None, uses validation_interval
    val_idx: Optional[List[int]] = None

    checkpoint_interval: int = 5000
    checkpoint_keep_last: int = 5
    checkpoint_best_metric: str = "valid/loss/total"
    checkpoint_best_mode: Literal["min", "max"] = "min"
    resume_from: Optional[Path] = None
    save_iters: Optional[List[int]] = None

    log_interval: int = 100
    wandb_enabled: bool = True
    wandb_project: str = "audioauth"
    wandb_entity: Optional[str] = None
    tensorboard_enabled: bool = True

    device: Literal["cuda", "cpu"] = "cuda"
    amp: bool = True
    compile_model: bool = False
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    use_distributed: bool = False
    dist_backend: Literal["nccl", "gloo"] = "nccl"
    dist_url: str = "env://"
    world_size: int = 1
    rank: int = 0
    gpu: Optional[int] = None
    local_rank: int = 0

    use_cloud_storage: bool = False
    cloud_provider: Literal["gcs", "s3"] = "gcs"
    cloud_bucket: Optional[str] = None
    cloud_prefix: Optional[str] = None

    detect_anomaly: bool = False
    profile: bool = False
    profile_steps: int = 10
    debug_gradients: bool = False

    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4


# --- Top-Level Config ---

class Config(BaseSettings):
    """Main configuration combining all sub-configs."""

    watermarking: WatermarkingConfig
    discriminator: DiscriminatorConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    metrics: MetricsConfig
    dataset: DatasetConfig
    run: RunConfig

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path, cli_args: Optional[List[str]] = None) -> "Config":
        """Load config from YAML file with optional CLI overrides."""
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        if cli_args:
            yaml_config = cls._apply_cli_overrides(yaml_config, cli_args)

        return cls(**yaml_config)

    def save_model_json(self, json_path: Path) -> None:
        """Save configuration as JSON file.

        Args:
            json_path: Path to save JSON config.
        """
        import json
        config_dict = self.model_dump()

        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj

        config_dict = convert_paths(config_dict)

        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_sources(
        cls,
        yaml_file: Optional[Path] = None,
        env_file: Optional[Path] = None,
        cli_args: Optional[List[str]] = None,
        **kwargs
    ) -> "Config":
        """Load config from multiple sources with priority: CLI > env > yaml > defaults."""
        config_dict = {}

        if yaml_file and yaml_file.exists():
            with open(yaml_file, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                config_dict = deep_update(config_dict, yaml_config)

        config_dict = deep_update(config_dict, kwargs)

        if cli_args:
            config_dict = cls._apply_cli_overrides(config_dict, cli_args)

        return cls(**config_dict)

    @staticmethod
    def _apply_cli_overrides(config_dict: Dict[str, Any], cli_args: List[str]) -> Dict[str, Any]:
        """Apply CLI overrides to config dictionary."""
        for arg in cli_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                keys = key.split(".")

                d = config_dict
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]

                try:
                    d[keys[-1]] = int(value)
                except ValueError:
                    try:
                        d[keys[-1]] = float(value)
                    except ValueError:
                        if value.lower() in ["true", "false"]:
                            d[keys[-1]] = value.lower() == "true"
                        else:
                            d[keys[-1]] = value

        return config_dict

    def save_to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def pretty_print(self) -> None:
        """Pretty print the configuration."""
        import json
        from pathlib import Path

        def convert_paths(obj):
            """Convert Path objects to strings for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        print("Configuration:")
        config_dict = convert_paths(self.model_dump())
        print(json.dumps(config_dict, indent=2))


def create_default_config() -> Config:
    """Create a default configuration.

    Note: This function is deprecated since all configuration values
    are now required. Please load configuration from a YAML file instead.

    Raises:
        NotImplementedError: Always raises since default configs are
            no longer supported.
    """
    raise NotImplementedError(
        "Default configurations are no longer supported. "
        "All configuration values must be explicitly provided via YAML. "
        "Please use Config.from_yaml() to load configuration from a file."
    )
