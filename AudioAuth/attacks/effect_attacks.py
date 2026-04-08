"""
Audio Effect Attacks Module

This module implements audio effect-based attacks for watermarked audio.
It provides various signal processing transformations that can be applied
to watermarked audio while maintaining ground truth alignment.
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torchaudio.functional as F
import julius
from julius import fft_conv1d

# julius.resample_frac is used instead of torchaudio.functional.resample_frac,
# which does not exist. All resampling in this module goes through julius.
from .effect_scheduler import EffectScheduler
from AudioAuth.exceptions import EffectSchedulerError, ParameterValidationError



def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm with PyTorch."""
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise

logger = logging.getLogger(__name__)

class AudioEffects:
    """
    Collection of audio effect transformations.
    All methods return (audio, mask) tuples.

    Attributes:
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio effects.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def apply_pink_noise(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        noise_std: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add pink noise to audio.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            noise_std: Standard deviation for pink noise

        Returns:
            Tuple of (noisy audio, unchanged mask)
        """
        noise = generate_pink_noise(audio.shape[-1]) * noise_std
        noise = noise.to(audio.device)
        noisy_audio = audio + noise.unsqueeze(0).unsqueeze(0).to(audio.device)
        return noisy_audio, mask

    def apply_lowpass_filter(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        cutoff_freq: float = 5000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a lowpass filter to the waveform.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            cutoff_freq: Cutoff frequency in Hz

        Returns:
            Tuple of (filtered audio, unchanged mask)
        """
        filtered = julius.lowpass_filter(audio, cutoff=cutoff_freq / self.sample_rate)
        return filtered, mask

    def apply_highpass_filter(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        cutoff_freq: float = 500
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a highpass filter to the waveform.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            cutoff_freq: Cutoff frequency in Hz

        Returns:
            Tuple of (filtered audio, unchanged mask)
        """
        filtered = julius.highpass_filter(audio, cutoff=cutoff_freq / self.sample_rate)
        return filtered, mask

    def apply_bandpass_filter(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a bandpass filter to the waveform by cascading
        a high-pass filter followed by a low-pass filter.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            cutoff_freq_low: Low cutoff frequency in Hz
            cutoff_freq_high: High cutoff frequency in Hz

        Returns:
            Tuple of (filtered audio, unchanged mask)
        """
        return julius.bandpass_filter(
            audio,
            cutoff_low=cutoff_freq_low / self.sample_rate,
            cutoff_high=cutoff_freq_high / self.sample_rate,
        ), mask

    def apply_volume_change(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        volume_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Change audio volume.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            volume_factor: Volume multiplication factor

        Returns:
            Tuple of (scaled audio, unchanged mask)
        """
        return audio * volume_factor, mask

    def apply_updown_resample(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        intermediate_freq: int = 32000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply upsample then downsample to audio.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            intermediate_freq: Intermediate frequency for resampling

        Returns:
            Tuple of (resampled audio, unchanged mask)
        """
        orig_shape = audio.shape
        audio_up = julius.resample_frac(audio, self.sample_rate, intermediate_freq)
        audio_down = julius.resample_frac(audio_up, intermediate_freq, self.sample_rate)

        # Resampling round-trips can shift length by a few samples; pad/truncate to match
        tmp = torch.zeros_like(audio)
        tmp[..., :audio_down.shape[-1]] = audio_down
        audio_down = tmp

        assert audio_down.shape == orig_shape, f"Shape mismatch: {audio_down.shape} vs {orig_shape}"
        return audio_down, mask

    def apply_echo(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        volume_range: Tuple[float, float] = (0.1, 0.5),
        duration_range: Tuple[float, float] = (0.1, 0.5)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply echo/reverb effect to audio.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            volume_range: Range for echo volume (min, max)
            duration_range: Range for echo duration in seconds (min, max)

        Returns:
            Tuple of (echoed audio, unchanged mask)
        """
        B, C, T = audio.shape

        volume = torch.FloatTensor(1).uniform_(*volume_range).item()
        duration = torch.FloatTensor(1).uniform_(*duration_range).item()

        n_samples = int(self.sample_rate * duration)

        impulse_response = torch.zeros(n_samples, dtype=audio.dtype, device=audio.device)
        impulse_response[0] = 1.0  # Direct sound

        if n_samples > 1:
            impulse_response[-1] = volume  # First reflection

        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        echoed_audio = fft_conv1d(audio, impulse_response)

        # Normalize to prevent clipping
        max_original = torch.max(torch.abs(audio))
        max_echoed = torch.max(torch.abs(echoed_audio))

        if max_echoed > 0:
            echoed_audio = echoed_audio * (max_original / max_echoed)

        # Convolution extends length; pad/truncate to match original
        tmp = torch.zeros_like(audio)
        tmp[..., :echoed_audio.shape[-1]] = echoed_audio
        echoed_audio = tmp

        return echoed_audio, mask


    def apply_boost_audio(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        amount: float = 20.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Boost audio volume by percentage.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            amount: Boost amount in percentage (0-100)

        Returns:
            Tuple of (boosted audio, unchanged mask)
        """
        boost_factor = 1 + amount / 100.0
        return audio * boost_factor, mask

    def apply_duck_audio(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        amount: float = 20.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Duck (reduce) audio volume by percentage.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            amount: Duck amount in percentage (0-100)

        Returns:
            Tuple of (ducked audio, unchanged mask)
        """
        duck_factor = 1 - amount / 100.0
        return audio * duck_factor, mask

    def apply_speed_change(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        speed_range: tuple = (0.5, 1.5)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Change the playback speed of audio using resampling.
        Note: This effect changes audio length temporarily but resamples back to original length.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            speed_range: Range for speed factor (min, max), <1.0 slows down, >1.0 speeds up

        Returns:
            Tuple of (speed-changed audio with original length, unchanged mask)
        """
        orig_shape = audio.shape
        speed_factor = torch.FloatTensor(1).uniform_(*speed_range).item()
        new_sample_rate = int(self.sample_rate / speed_factor)

        resampled_audio = julius.resample_frac(audio, self.sample_rate, new_sample_rate)

        resampled_back = julius.resample_frac(resampled_audio, new_sample_rate, self.sample_rate)

        # Resampling round-trips can shift length by a few samples; pad/truncate to match
        tmp = torch.zeros_like(audio)
        tmp[..., :resampled_back.shape[-1]] = resampled_back
        resampled_back = tmp

        assert resampled_back.shape == orig_shape, f"Shape mismatch: {resampled_back.shape} vs {orig_shape}"

        return resampled_back, mask

    def apply_random_noise(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        noise_std: float = 0.001
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to audio.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Tuple of (noisy audio, unchanged mask)
        """
        noise = torch.randn_like(audio) * noise_std
        noisy_audio = audio + noise
        return noisy_audio, mask


    def apply_smooth(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        window_size_range: tuple = (2, 10)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Smooth the input audio using a moving average filter.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            window_size_range: Range for moving average window size (min, max)

        Returns:
            Tuple of (smoothed audio, unchanged mask)
        """
        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
        kernel = torch.ones(1, 1, window_size).type(audio.type()) / window_size
        kernel = kernel.to(audio.device)

        smoothed = fft_conv1d(audio, kernel)

        # Convolution extends length; pad/truncate to match original
        tmp = torch.zeros_like(audio)
        tmp[..., :smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return smoothed, mask

    def apply_mp3_compression(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        bitrate: str = "128k"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MP3 compression/decompression round-trip via ffmpeg.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            bitrate: MP3 bitrate (e.g., '32k', '64k', '128k', '320k')

        Returns:
            Tuple of (compressed audio, unchanged mask)
        """
        return self._codec_roundtrip(audio, mask, codec="libmp3lame", ext="mp3", bitrate=bitrate)

    def apply_aac_compression(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        bitrate: str = "64k"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply AAC compression/decompression round-trip via ffmpeg.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            bitrate: AAC bitrate (e.g., '32k', '64k', '128k')

        Returns:
            Tuple of (compressed audio, unchanged mask)
        """
        return self._codec_roundtrip(audio, mask, codec="aac", ext="m4a", bitrate=bitrate)

    def _codec_roundtrip(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        codec: str,
        ext: str,
        bitrate: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to a lossy codec and decode back via ffmpeg subprocess.

        Uses straight-through estimation: the compressed audio replaces the
        original in the forward pass, but gradients flow through unchanged.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            codec: ffmpeg codec name (e.g., 'libmp3lame', 'aac')
            ext: Output file extension (e.g., 'mp3', 'm4a')
            bitrate: Codec bitrate string

        Returns:
            Tuple of (compressed audio, unchanged mask)
        """
        device = audio.device
        orig_shape = audio.shape
        B, C, T = orig_shape

        compressed_batches = []
        for b in range(B):
            wav = audio[b]  # [C, T]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f_in, \
                 tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as f_enc, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f_out:

                torchaudio.save(f_in.name, wav.cpu(), self.sample_rate)

                subprocess.run(
                    ["ffmpeg", "-y", "-i", f_in.name, "-c:a", codec, "-b:a", bitrate, f_enc.name],
                    capture_output=True, check=True
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", f_enc.name, "-c:a", "pcm_s16le", "-ar", str(self.sample_rate), f_out.name],
                    capture_output=True, check=True
                )
                decoded, sr = torchaudio.load(f_out.name)
                decoded = decoded.to(device)

            compressed_batches.append(decoded)

        result = torch.stack(compressed_batches, dim=0)  # [B, C, T']

        if result.shape[-1] > T:
            result = result[..., :T]
        elif result.shape[-1] < T:
            pad_len = T - result.shape[-1]
            result = torch.nn.functional.pad(result, (0, pad_len))

        # Straight-through: attach gradients from original
        if audio.requires_grad:
            result = audio + (result - audio).detach()

        return result, mask

    def apply_encodec_compression(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor,
        n_quantizers: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply EnCodec compression/decompression round-trip.

        Uses the pretrained EnCodec model to encode audio to discrete codes
        and decode back, simulating neural codec compression.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]
            n_quantizers: Number of RVQ quantizers (default 16, as in Table II)

        Returns:
            Tuple of (compressed audio, unchanged mask)
        """
        try:
            from encodec import EncodecModel
        except ImportError:
            logger.warning("encodec package not installed, skipping EnCodec compression")
            return audio, mask

        device = audio.device
        orig_shape = audio.shape
        T = orig_shape[-1]

        if not hasattr(self, '_encodec_model') or self._encodec_model is None:
            if self.sample_rate == 24000:
                self._encodec_model = EncodecModel.encodec_model_24khz()
            else:
                self._encodec_model = EncodecModel.encodec_model_48khz()
            self._encodec_model.set_target_bandwidth(6.0)
            self._encodec_model.to(device)
            self._encodec_model.eval()

        model = self._encodec_model

        with torch.no_grad():
            model_sr = model.sample_rate
            if self.sample_rate != model_sr:
                audio_resampled = julius.resample_frac(audio, self.sample_rate, model_sr)
            else:
                audio_resampled = audio

            encoded_frames = model.encode(audio_resampled)

            codes = encoded_frames[0]  # (codes, scale)
            if isinstance(codes, tuple):
                codes, scale = codes
            else:
                scale = None

            if codes.shape[1] > n_quantizers:
                codes = codes[:, :n_quantizers, :]

            if scale is not None:
                encoded_frames = [((codes, scale),)]
            else:
                encoded_frames = [(codes,)]

            decoded = model.decode(encoded_frames)

            if self.sample_rate != model_sr:
                decoded = julius.resample_frac(decoded, model_sr, self.sample_rate)

        if decoded.shape[-1] > T:
            decoded = decoded[..., :T]
        elif decoded.shape[-1] < T:
            pad_len = T - decoded.shape[-1]
            decoded = torch.nn.functional.pad(decoded, (0, pad_len))

        # Straight-through estimation
        if audio.requires_grad:
            decoded = audio + (decoded - audio).detach()

        return decoded, mask

    def apply_identity(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identity effect -- returns audio unchanged.
        Useful as a baseline/control for testing.

        Args:
            audio: Input audio tensor [B, C, T]
            mask: Binary mask tensor [B, C, T]

        Returns:
            Tuple of (unchanged audio, unchanged mask)
        """
        return audio, mask



class EffectAttacks(nn.Module):
    """
    PyTorch module for applying audio effects as attacks.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        audio_effects: AudioEffects instance providing the effect implementations.
        all_effect_names: Complete list of recognized effect names.
        effect_enabled: Dict mapping effect name to enabled flag.
        effect_names: List of currently enabled effect names.
        effect_params: Dict of parameter configurations for each effect.
        scheduler: EffectScheduler for adaptive effect/parameter selection.
        compound_chain_prob: Probability of applying a compound chain of effects
            (Section III-B: compound effect chains for robustness training).
        max_chain_length: Maximum number of effects in a compound chain.
        length_preserving_effects: Set of effects that do not change audio length.
        stats: Per-effect usage counts plus a 'total' key.
        effect_distribution: Per-effect cumulative usage counts for monitoring.
        total_samples_processed: Running total of processed samples.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        effect_enabled: Dict[str, bool] = None,  # Required parameter
        effect_params: Optional[Dict[str, Any]] = None,
        scheduler_config: Dict[str, Any] = None  # Required scheduler configuration
    ) -> None:
        """
        Initialize EffectAttacks with adaptive selection using EffectScheduler.

        Args:
            sample_rate: Audio sample rate in Hz
            effect_enabled: Dict of enabled flags for each effect (required)
            effect_params: Dict of parameters for each effect with choices format (optional)
            scheduler_config: Configuration for EffectScheduler (required)
        """
        super().__init__()

        self.sample_rate = sample_rate

        self.audio_effects = AudioEffects(sample_rate)

        self.all_effect_names = [
            'pink_noise', 'lowpass',
            'highpass', 'bandpass', 'volume', 'identity',
            'updown_resample', 'echo', 'boost_audio',
            'duck_audio', 'speed', 'random_noise', 'smooth',
            'mp3_compression', 'aac_compression', 'encodec_compression'
        ]

        if effect_enabled is None:
            raise ValueError("effect_enabled parameter is required")
        if scheduler_config is None:
            raise ValueError("scheduler_config parameter is required")

        self.effect_enabled = effect_enabled

        # Missing effects are treated as disabled
        self.effect_names = [name for name in self.all_effect_names if effect_enabled.get(name, False)]

        if not self.effect_names:
            raise ValueError("At least one effect must be enabled")

        self.effect_params = effect_params or {}

        # Filter effect_params to only include enabled effects
        enabled_effect_params = {}
        for name in self.effect_names:
            params = self.effect_params.get(name, {})
            if params:
                enabled_effect_params[name] = params
            else:
                enabled_effect_params[name] = {}

        self.scheduler = EffectScheduler(
            effect_params=enabled_effect_params,
            beta=scheduler_config.get('beta', 0.9),
            ber_threshold=scheduler_config.get('ber_threshold', 0.001),
            miou_threshold=scheduler_config.get('miou_threshold', 0.95),
            temperature_start=scheduler_config.get('temperature_start', 1.0),
            temperature_end=scheduler_config.get('temperature_end', 0.7)
        )

        # Compound effect chain configuration (Section III-B: multiple effects
        # applied sequentially to a single batch for harder robustness training)
        self.compound_chain_prob = scheduler_config.get('compound_chain_prob', 0.0)
        self.max_chain_length = scheduler_config.get('max_chain_length', 3)

        # Effects whose output length always equals input length (no resampling or convolution)
        self.length_preserving_effects = {
            'white_noise', 'pink_noise',
            'lowpass', 'highpass', 'bandpass', 'volume', 'identity'
        }

        self.stats = {name: 0 for name in self.effect_names}
        self.stats['total'] = 0

        self.effect_distribution = {name: 0 for name in self.effect_names}
        self.total_samples_processed = 0


    def get_effect_distribution(self) -> Dict[str, float]:
        """
        Get the actual effect distribution from processed samples.

        Returns:
            Dictionary with effect names and their actual usage percentages
        """
        if self.total_samples_processed == 0:
            return {name: 0.0 for name in self.effect_names}

        distribution = {}
        for name in self.effect_names:
            count = self.effect_distribution.get(name, 0)
            distribution[name] = count / self.total_samples_processed

        return distribution

    def reset_distribution_tracking(self) -> None:
        """Reset effect distribution tracking counters."""
        self.effect_distribution = {name: 0 for name in self.effect_names}
        self.total_samples_processed = 0
        logger.debug("Reset effect distribution tracking")

    def _select_effects(self, batch_size: int) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select effects for each batch item using scheduler.

        Args:
            batch_size: Number of items in batch

        Returns:
            List of tuples (effect_name, selected_params) for each batch item
        """
        selected_effects = self.scheduler.select_effects(batch_size)
        return selected_effects

    def _get_effect_function(self, effect_name: str, selected_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get effect function by name with configured parameters.

        Args:
            effect_name: Name of the effect
            selected_params: Optional pre-selected parameters from scheduler

        Returns:
            Effect function
        """
        if selected_params is not None:
            params = selected_params
        else:
            params = self.effect_params.get(effect_name, {})

        if effect_name == 'white_noise':
            snr_db = params.get('snr_db', 20.0) if isinstance(params, dict) else 20.0
            return lambda x, m: self.audio_effects.apply_white_noise(x, m, snr_db=snr_db)

        elif effect_name == 'pink_noise':
            noise_std = params.get('noise_std', 0.01) if isinstance(params, dict) else 0.01
            return lambda x, m: self.audio_effects.apply_pink_noise(x, m, noise_std=noise_std)

        elif effect_name == 'lowpass':
            cutoff = params.get('cutoff_freq', 4000) if isinstance(params, dict) else 4000
            return lambda x, m: self.audio_effects.apply_lowpass_filter(x, m, cutoff_freq=cutoff)

        elif effect_name == 'highpass':
            cutoff = params.get('cutoff_freq', 1000) if isinstance(params, dict) else 1000
            return lambda x, m: self.audio_effects.apply_highpass_filter(x, m, cutoff_freq=cutoff)

        elif effect_name == 'bandpass':
            if isinstance(params, dict):
                if 'frequency_pairs' in params and 'choices' in params['frequency_pairs']:
                    freq_choices = params['frequency_pairs']['choices']
                    if freq_choices and len(freq_choices) > 0:
                        freq_pair = freq_choices[0]
                        low = freq_pair[0] if len(freq_pair) > 0 else 300
                        high = freq_pair[1] if len(freq_pair) > 1 else 3400
                    else:
                        low, high = 300, 3400
                elif '_frequency_pair' in params:
                    freq_pair = params['_frequency_pair']
                    low = freq_pair[0] if len(freq_pair) > 0 else 300
                    high = freq_pair[1] if len(freq_pair) > 1 else 3400
                else:
                    low = params.get('low_cutoff', 300)
                    high = params.get('high_cutoff', 3400)
            else:
                low, high = 300, 3400
            return lambda x, m: self.audio_effects.apply_bandpass_filter(x, m, cutoff_freq_low=low, cutoff_freq_high=high)

        elif effect_name == 'volume':
            factor = params.get('volume_factor', 0.8) if isinstance(params, dict) else 0.8
            return lambda x, m: self.audio_effects.apply_volume_change(x, m, volume_factor=factor)

        elif effect_name == 'updown_resample':
            intermediate_freq = params.get('intermediate_freq', 32000) if isinstance(params, dict) else 32000
            return lambda x, m: self.audio_effects.apply_updown_resample(x, m, intermediate_freq=intermediate_freq)

        elif effect_name == 'echo':
            if isinstance(params, dict):
                volume_range = params.get('volume_range', (0.1, 0.5))
                duration_range = params.get('duration_range', (0.1, 0.5))
            else:
                volume_range = (0.1, 0.5)
                duration_range = (0.1, 0.5)
            return lambda x, m: self.audio_effects.apply_echo(x, m, volume_range=volume_range, duration_range=duration_range)


        elif effect_name == 'boost_audio':
            amount = params.get('amount', 20.0) if isinstance(params, dict) else 20.0
            return lambda x, m: self.audio_effects.apply_boost_audio(x, m, amount=amount)

        elif effect_name == 'duck_audio':
            amount = params.get('amount', 20.0) if isinstance(params, dict) else 20.0
            return lambda x, m: self.audio_effects.apply_duck_audio(x, m, amount=amount)

        elif effect_name == 'speed':
            speed_range = params.get('speed_range', (0.5, 1.5)) if isinstance(params, dict) else (0.5, 1.5)
            return lambda x, m: self.audio_effects.apply_speed_change(x, m, speed_range=speed_range)

        elif effect_name == 'random_noise':
            noise_std = params.get('noise_std', 0.001) if isinstance(params, dict) else 0.001
            return lambda x, m: self.audio_effects.apply_random_noise(x, m, noise_std=noise_std)


        elif effect_name == 'smooth':
            window_size_range = params.get('window_size_range', (2, 10)) if isinstance(params, dict) else (2, 10)
            return lambda x, m: self.audio_effects.apply_smooth(x, m, window_size_range=window_size_range)

        elif effect_name == 'identity':
            return lambda x, m: self.audio_effects.apply_identity(x, m)

        elif effect_name == 'mp3_compression':
            bitrate = params.get('bitrate', '128k') if isinstance(params, dict) else '128k'
            return lambda x, m: self.audio_effects.apply_mp3_compression(x, m, bitrate=bitrate)

        elif effect_name == 'aac_compression':
            bitrate = params.get('bitrate', '64k') if isinstance(params, dict) else '64k'
            return lambda x, m: self.audio_effects.apply_aac_compression(x, m, bitrate=bitrate)

        elif effect_name == 'encodec_compression':
            n_quantizers = params.get('n_quantizers', 16) if isinstance(params, dict) else 16
            return lambda x, m: self.audio_effects.apply_encodec_compression(x, m, n_quantizers=n_quantizers)

        else:
            logger.warning(f"Unknown effect: {effect_name}, using identity")
            return lambda x, m: self.audio_effects.apply_identity(x, m)

    def forward(
        self,
        watermarked: torch.Tensor,
        ground_truth_presence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Apply one audio effect to the entire batch.

        Args:
            watermarked: Watermarked audio tensor [B, C, T]
            ground_truth_presence: Binary mask [B, C, T]

        Returns:
            Tuple of:
                - Modified audio tensor [B, C, T] with exact original length preserved
                - Modified ground truth mask [B, C, T] with exact original length preserved
                - Statistics dictionary
        """
        original_shape = watermarked.shape
        batch_size, num_channels, num_samples = watermarked.shape

        # Determine if we apply a compound chain (Section III-B: sequentially
        # applying multiple effects to stress-test watermark robustness)
        use_compound = (
            self.compound_chain_prob > 0 and
            np.random.random() < self.compound_chain_prob
        )

        if use_compound:
            chain_length = np.random.randint(2, self.max_chain_length + 1)
            effect_selections = self._select_effects(chain_length)
        else:
            effect_selections = self._select_effects(1)

        effects_applied = [name for name, _ in effect_selections]
        effect_params_used = [(name, params) for name, params in effect_selections]

        attacked = watermarked
        mask = ground_truth_presence

        for effect_name, selected_params in effect_selections:
            try:
                effect_func = self._get_effect_function(effect_name, selected_params)

                if effect_func:
                    attacked, mask = effect_func(attacked, mask)

                    # Length safety: some effects (echo, resample, smooth) may change length
                    if attacked.shape[-1] != num_samples:
                        logger.warning(f"Effect {effect_name} changed audio length from {num_samples} to {attacked.shape[-1]}. Fixing...")
                        if attacked.shape[-1] > num_samples:
                            attacked = attacked[..., :num_samples]
                            mask = mask[..., :num_samples]
                        else:
                            pad_len = num_samples - attacked.shape[-1]
                            attacked = torch.nn.functional.pad(attacked, (0, pad_len))
                            mask = torch.nn.functional.pad(mask, (0, pad_len))

                    self.stats[effect_name] += batch_size
                    self.effect_distribution[effect_name] += batch_size
                    self.total_samples_processed += batch_size

                else:
                    logger.warning(f"Effect function not found for {effect_name}, skipping")

            except (EffectSchedulerError, ParameterValidationError) as e:
                logger.error(f"Critical scheduler error for effect {effect_name}: {e}")
                raise
            except Exception as e:
                logger.warning(f"Non-critical error applying effect {effect_name}: {e}")

        assert attacked.shape == original_shape, \
            f"Shape mismatch after effects: {attacked.shape} vs {original_shape}"
        assert mask.shape == original_shape, \
            f"Mask shape mismatch after effects: {mask.shape} vs {original_shape}"

        self.stats['total'] += 1

        stats_dict = {
            'effects_applied': effects_applied,
            'effect_params_used': effect_params_used,
            'effect_stats': self.stats.copy(),
            'original_shape': original_shape,
            'final_shape': attacked.shape,
            'compound_chain': use_compound
        }

        return attacked, mask, stats_dict
