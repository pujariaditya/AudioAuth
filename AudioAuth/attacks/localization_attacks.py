"""
Audio watermark localization attack module.

This module provides functionality for attacking watermarked audio samples and generating
corresponding ground truth labels for watermark presence detection. It applies various
attack techniques to simulate real-world scenarios where watermarked audio might
be manipulated or corrupted.
"""

import logging
import os
import sys
from typing import Tuple, Dict, List, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .ste import STEMaskGenerator, create_ste_mask

# Only constants used for demo/testing, not for actual attack logic
NOISE_AMPLITUDE = 0.0001  # 0.01% noise for watermarking simulation (demo only)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TEST_DURATION = 5

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class LocalizationAttacks(nn.Module):
    """
    PyTorch module for attacking watermarked audio with localization ground truth.

    This module applies various attack techniques to watermarked audio samples
    targeting approximately 20% of audio segments for manipulation. It generates
    corresponding ground truth labels indicating watermark presence.

    Attributes:
        sample_rate: Audio sampling rate in Hz.
        window_duration: Duration of each segment window in seconds.
        segment_length: Number of samples per segment.
        target_ratio: Fraction of segments to attack per batch item.
        original_revert_prob: Probability of reverting a segment to original audio.
        zero_replace_prob: Probability of replacing a segment with silence.
        ste_mask_generator: STEMaskGenerator (Straight-Through Estimator) for creating
            binary attack masks that allow gradient flow during training -- the
            forward pass uses hard binary decisions while the backward pass
            passes gradients through unchanged.
        stats: Per-attack-type sample counts, reset each forward call.
    """

    def __init__(
        self,
        sample_rate: int,
        window_duration: float,
        target_ratio: float,
        original_revert_prob: float,
        zero_replace_prob: float
    ) -> None:
        """
        Initialize the LocalizationAttacks module.

        Args:
            sample_rate: Sampling rate of the audio data in Hz
            window_duration: Duration of the window in seconds for each segment
            target_ratio: Target ratio of segments to attack (0-1)
            original_revert_prob: Probability of reverting to original
            zero_replace_prob: Probability of replacing with zeros

        Raises:
            ValueError: If sample_rate <= 0 or window_duration <= 0 or probabilities invalid
        """
        super().__init__()

        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if window_duration <= 0:
            raise ValueError(f"Window duration must be positive, got {window_duration}")
        if not 0 <= target_ratio <= 1:
            raise ValueError(f"Target ratio must be between 0 and 1, got {target_ratio}")

        prob_sum = original_revert_prob + zero_replace_prob
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Probabilities must sum to ~1.0, got {prob_sum:.3f}")

        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.segment_length = int(sample_rate * window_duration)

        self.target_ratio = target_ratio
        self.original_revert_prob = original_revert_prob
        self.zero_replace_prob = zero_replace_prob

        # STE enables gradient flow through the binary attack/no-attack decision:
        # the forward pass produces hard 0/1 masks, while the backward pass
        # treats the thresholding as an identity function.
        self.ste_mask_generator = STEMaskGenerator()

        self._reset_stats()

    def _reset_stats(self) -> None:
        """Reset attack statistics to zero."""
        self.stats = {
            'original_revert': 0,
            'zero_replace': 0,
            'unchanged': 0
        }

    def _apply_original_revert(
        self,
        watermarked: torch.Tensor,
        original: torch.Tensor,
        update_original: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
        start: int,
        end: int
    ) -> None:
        """
        Revert watermarked audio segment to original audio.

        Args:
            watermarked: Watermarked audio tensor to modify
            original: Original audio tensor to copy from
            update_original: Updated original tensor to modify
            ground_truth: Ground truth presence tensor to update
            batch_idx: Batch index
            start: Start sample index
            end: End sample index
        """
        watermarked[batch_idx, :, start:end] = original[batch_idx, :, start:end]
        update_original[batch_idx, :, start:end] = original[batch_idx, :, start:end]
        # ground_truth marking is handled in forward() with STE
        self.stats['original_revert'] += end - start
        logger.debug(f"Applied original revert to batch {batch_idx}, samples {start}:{end}")

    def _apply_zero_replace(
        self,
        watermarked: torch.Tensor,
        update_original: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
        start: int,
        end: int
    ) -> None:
        """
        Replace audio segment with zeros.

        Args:
            watermarked: Watermarked audio tensor to modify
            update_original: Updated original tensor to modify
            ground_truth: Ground truth presence tensor to update
            batch_idx: Batch index
            start: Start sample index
            end: End sample index
        """
        watermarked[batch_idx, :, start:end] = 0
        update_original[batch_idx, :, start:end] = 0
        # ground_truth marking is handled in forward() with STE
        self.stats['zero_replace'] += end - start
        logger.debug(f"Applied zero replacement to batch {batch_idx}, samples {start}:{end}")


    def forward(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Apply random attacks to watermarked audio and generate ground truth.

        This method processes watermarked audio by applying various attack
        techniques to approximately 20% of the segments. Each segment can be:
        - Reverted to original audio (removing watermark)
        - Replaced with zeros

        Args:
            original: Original audio batch tensor.
                     Shape: [batch_size, num_channels, num_samples]
            watermarked: Watermarked audio batch tensor.
                        Shape: [batch_size, num_channels, num_samples]

        Returns:
            Tuple containing:
                - torch.Tensor: Attacked watermarked audio [B, C, T]
                - torch.Tensor: Ground truth presence [B, 1, T] (1=present, 0=absent)
                - torch.Tensor: Updated original audio reflecting modifications [B, C, T]
                - Dict[str, float]: Statistics of attack types applied (percentages)

        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        if original.shape != watermarked.shape:
            raise ValueError(f"Shape mismatch: original {original.shape} != watermarked {watermarked.shape}")

        original = original.clone().float()
        update_original = original.clone()
        watermarked = watermarked.clone().float()

        batch_size, num_channels, num_samples = watermarked.shape

        device = watermarked.device
        attack_probability_mask = torch.zeros((batch_size, 1, num_samples), dtype=torch.float32, device=device)
        ground_truth_presence = torch.ones((batch_size, 1, num_samples), dtype=torch.float32, device=device)

        self._reset_stats()
        total_samples = batch_size * num_samples

        total_segments = int(np.ceil(num_samples / self.segment_length))
        segments_to_modify = int(total_segments * self.target_ratio)

        for batch_idx in range(batch_size):
            available_starts = np.arange(0, num_samples, self.segment_length)

            start_points = np.random.choice(
                available_starts,
                segments_to_modify,
                replace=False
            )

            for start in start_points:
                end = min(start + self.segment_length, num_samples)
                probability = np.random.rand()

                if probability < self.original_revert_prob:
                    self._apply_original_revert(
                        watermarked, original, update_original, ground_truth_presence,
                        batch_idx, start, end
                    )
                    attack_probability_mask[batch_idx, 0, start:end] = 1.0
                else:
                    self._apply_zero_replace(
                        watermarked, update_original, ground_truth_presence,
                        batch_idx, start, end
                    )
                    attack_probability_mask[batch_idx, 0, start:end] = 1.0

        modified_samples = sum([
            self.stats['original_revert'],
            self.stats['zero_replace']
        ])
        self.stats['unchanged'] = total_samples - modified_samples

        for key in self.stats:
            self.stats[key] = float((self.stats[key] / total_samples) * 100)

        # STE creates a binary mask with gradient flow: hard 0/1 forward,
        # identity gradient backward (see ste.py STEBinarize).
        if watermarked.requires_grad:
            attack_mask_with_gradient = self.ste_mask_generator.mask_module(attack_probability_mask)
            ground_truth_presence = 1.0 - attack_mask_with_gradient
        else:
            ground_truth_presence = 1.0 - attack_probability_mask

        ground_truth_presence = ground_truth_presence.float()

        return (
            watermarked,
            ground_truth_presence,
            update_original,
            self.stats
        )

# --- Helper Functions ---

def load_and_preprocess_audio(
    file_path: str,
    target_sample_rate: int,
    target_duration: float
) -> torch.Tensor:
    """
    Load and preprocess audio file.

    Args:
        file_path: Path to audio file
        target_sample_rate: Target sampling rate in Hz
        target_duration: Target duration in seconds

    Returns:
        Preprocessed audio tensor with shape [1, num_samples]

    Raises:
        IOError: If file cannot be loaded
        ValueError: If audio processing fails
    """
    try:
        waveform, original_sr = torchaudio.load(file_path)
        logger.info(f"Loaded audio: {file_path}, original_sr={original_sr}Hz, shape={waveform.shape}")

        if original_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(original_sr, target_sample_rate)
            waveform = resampler(waveform)
            logger.debug(f"Resampled from {original_sr}Hz to {target_sample_rate}Hz")

        samples_to_keep = int(target_duration * target_sample_rate)
        waveform = waveform[:, :samples_to_keep]

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug("Converted stereo to mono")

        return waveform

    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}", exc_info=True)
        raise IOError(f"Failed to load audio file: {file_path}") from e

# --- Demo ---

def main() -> None:
    """
    Main function to demonstrate LocalizationAttacks functionality.

    This function:
    1. Loads audio files from the audio_samples directory
    2. Creates watermarked versions by adding noise
    3. Applies attacks using LocalizationAttacks

    Raises:
        ValueError: If insufficient audio files are found
        IOError: If file operations fail
    """
    try:
        logger.info("Starting localization attack demonstration")

        attacker = LocalizationAttacks(DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_DURATION)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        audio_folder = os.path.join(parent_dir, "audio_samples")

        logger.info(f"Looking for audio files in: {audio_folder}")

        audio_files = [f for f in os.listdir(audio_folder)
                      if f.endswith(('.wav', '.mp3'))][:2]

        if len(audio_files) < 2:
            raise ValueError(f"Need at least 2 audio files in {audio_folder}, found {len(audio_files)}")

        logger.info(f"Found {len(audio_files)} audio files: {audio_files}")

        audio_tensors = []
        for audio_file in audio_files:
            file_path = os.path.join(audio_folder, audio_file)
            waveform = load_and_preprocess_audio(
                file_path,
                DEFAULT_SAMPLE_RATE,
                DEFAULT_TEST_DURATION
            )
            audio_tensors.append(waveform)

        original_batch = torch.stack(audio_tensors)

        noise = torch.randn_like(original_batch) * NOISE_AMPLITUDE
        watermarked_batch = original_batch + noise

        logger.info("Created watermarked audio by adding noise")

        attacked_batch, ground_truth, updated_original, stats = attacker(
            original_batch,
            watermarked_batch
        )

        logger.info("Attack Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}%")

        logger.info("Localization attack demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
