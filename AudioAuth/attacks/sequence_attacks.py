"""Sequence-based attacks for watermark robustness evaluation.

Applies attacks to sequential segments of audio with configurable parameters.
Supported attacks include reversing, head/tail trimming, crop replacement,
segment shuffling, and chunk shuffling.
"""

import logging
import os
import sys
from typing import Dict, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Only constants used for demo/testing, not for actual attack logic
NOISE_LEVEL = 0.0001  # noise level for watermarking simulation (demo only)

class SequenceAttacks(nn.Module):
    """
    A PyTorch module for attacking watermarked audio samples with sequence transformations.

    This module applies various sequence-level attacks to improve the robustness of
    watermark decoding, including realistic cropping, time stretching, and shuffling.

    Args:
        sample_rate (int): Sampling rate of the audio data in Hz.
        reverse_prob (float): Probability of reversing audio
        head_trim_prob (float): Probability of trimming samples from head
        tail_trim_prob (float): Probability of trimming samples from tail
        crop_replacement_prob (float): Probability of crop replacement (replaces circular shift)
        shuffle_prob (float): Probability of shuffling segments
        chunk_shuffle_prob (float): Probability of chunk shuffle
        segment_duration (float): Duration for shuffle segments (in seconds)
        chunk_divisions (int): Number of chunks for chunk operations
        max_trim_ms (float): Maximum trim duration in milliseconds

    Attributes:
        sample_rate (int): The audio sampling rate.
        unchanged_prob (float): Residual probability of applying no attack (1 - sum of all probs).
        reverse_prob (float): Probability of reversing audio.
        head_trim_prob (float): Probability of trimming from head.
        tail_trim_prob (float): Probability of trimming from tail.
        crop_replacement_prob (float): Probability of crop replacement.
        shuffle_prob (float): Probability of segment shuffling.
        chunk_shuffle_prob (float): Probability of chunk shuffling.
        segment_duration (float): Duration per shuffle segment in seconds.
        chunk_divisions (int): Number of chunk divisions.
        max_trim_ms (float): Maximum trim duration in milliseconds.
        max_trim_samples (int): Maximum trim in samples (derived from max_trim_ms).
        methods (List[str]): List of enabled attack methods.
        stats (Dict[str, float]): Statistics tracking attack usage.

    Raises:
        ValueError: If invalid attack methods are provided.
    """
    def __init__(
        self,
        sample_rate: int,
        reverse_prob: float,
        head_trim_prob: float,
        tail_trim_prob: float,
        crop_replacement_prob: float,
        shuffle_prob: float,
        chunk_shuffle_prob: float,
        segment_duration: float,
        chunk_divisions: int,
        max_trim_ms: float = 12.0
    ) -> None:
        """
        Initialize the SequenceAttacks module.

        Args:
            sample_rate (int): Sampling rate of the audio data in Hz.
            reverse_prob: Probability of reversing audio
            head_trim_prob: Probability of trimming samples from head (realistic cropping)
            tail_trim_prob: Probability of trimming samples from tail (realistic cropping)
            crop_replacement_prob: Probability of crop replacement (replaces circular shift)
            shuffle_prob: Probability of shuffling segments
            chunk_shuffle_prob: Probability of chunk shuffle
            segment_duration: Duration for shuffle segments (in seconds)
            chunk_divisions: Number of chunks for chunk shuffle
            max_trim_ms: Maximum trim duration in milliseconds (default: 12ms)

        Raises:
            ValueError: If sample_rate is not positive or if invalid methods are provided.
        """
        super().__init__()

        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

        # Any remaining probability goes to 'unchanged' case
        prob_sum = reverse_prob + head_trim_prob + tail_trim_prob + crop_replacement_prob + shuffle_prob + chunk_shuffle_prob
        if prob_sum > 1.0:
            raise ValueError(f"Attack probabilities cannot sum to more than 1.0, got {prob_sum:.3f}")
        if prob_sum < 0.0:
            raise ValueError(f"Attack probabilities must be non-negative, got {prob_sum:.3f}")

        self.unchanged_prob = max(0.0, 1.0 - prob_sum)

        self.sample_rate = sample_rate

        self.reverse_prob = reverse_prob
        self.head_trim_prob = head_trim_prob
        self.tail_trim_prob = tail_trim_prob
        self.crop_replacement_prob = crop_replacement_prob
        self.shuffle_prob = shuffle_prob
        self.chunk_shuffle_prob = chunk_shuffle_prob
        self.segment_duration = segment_duration
        self.chunk_divisions = chunk_divisions
        self.max_trim_ms = max_trim_ms
        self.max_trim_samples = int(max_trim_ms * sample_rate / 1000)

        self.methods = ['reverse', 'head_trim', 'tail_trim', 'crop_replacement', 'shuffle', 'chunk_shuffle']

        self.stats = {method: 0 for method in self.methods}
        self.stats['unchanged'] = 0

    def forward(
        self,
        updated_original: torch.Tensor,
        watermarked: torch.Tensor,
        ground_truth_presence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], str]:
        """
        Apply the same attack to all samples in the batch.

        This method applies a single attack type to the entire batch, ensuring
        consistency across all samples. The attack is also applied to the original
        audio and ground truth presence indicators to maintain alignment.

        Current implementation applies one method per batch for deterministic
        training. For per-sample variety, modify the method selection logic.

        Args:
            updated_original (torch.Tensor): Original audio batch.
                Shape: [batch_size, num_channels, num_samples]
                Device: Any (tensors must be on the same device)
                Dtype: Float32 recommended
            watermarked (torch.Tensor): Watermarked audio batch.
                Shape: [batch_size, num_channels, num_samples]
                Device: Must match updated_original
                Dtype: Must match updated_original
            ground_truth_presence (torch.Tensor): Ground truth presence indicators.
                Shape: [batch_size, num_channels, num_samples]
                Device: Must match other tensors
                Dtype: Float32 with values 0.0 (absent) or 1.0 (present)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], str]:
                - attacked_watermarked (torch.Tensor): Attacked watermarked audio tensor
                - updated_original (torch.Tensor): Attacked original audio
                - ground_truth_presence (torch.Tensor): Attacked ground truth presence
                - stats (Dict[str, float]): Attack statistics as percentages (one method = 100%)
                - method (str): The attack method applied

        Raises:
            RuntimeError: If tensor operations fail.
            ValueError: If input tensors have incompatible shapes or invalid dimensions.

        Example:
            >>> attacker = SequenceAttacks(sample_rate=16000, head_trim_prob=0.5,
            ...                         tail_trim_prob=0.3, max_trim_ms=12.0, **other_params)
            >>> attacked, orig, gt, stats, method = attacker(original, watermarked, gt_presence
        """
        try:
            if not (updated_original.shape == watermarked.shape == ground_truth_presence.shape):
                raise ValueError(
                    f"Input tensors must have the same shape. Got: "
                    f"updated_original={updated_original.shape}, "
                    f"watermarked={watermarked.shape}, "
                    f"ground_truth_presence={ground_truth_presence.shape}"
                )

            if len(watermarked.shape) != 3:
                raise ValueError(f"Input tensors must be 3D [batch, channels, samples], got shape: {watermarked.shape}")

            batch_size, num_channels, num_samples = watermarked.shape

            if num_samples < 100:  # Less than ~6ms at 16kHz
                logger.warning(f"Audio very short ({num_samples} samples). Some attacks may be disabled.")

            if batch_size <= 0:
                raise ValueError(f"Batch size must be positive, got: {batch_size}")

            updated_original = updated_original.clone()
            watermarked = watermarked.clone()
            ground_truth_presence = ground_truth_presence.clone()

            self.stats = {k: 0 for k in self.stats}
            self.stats['unchanged'] = 0

            random_value = torch.rand(1).item()

            cumulative_prob = 0.0
            if random_value < (cumulative_prob := cumulative_prob + self.reverse_prob):
                method = 'reverse'
            elif random_value < (cumulative_prob := cumulative_prob + self.head_trim_prob):
                method = 'head_trim'
            elif random_value < (cumulative_prob := cumulative_prob + self.tail_trim_prob):
                method = 'tail_trim'
            elif random_value < (cumulative_prob := cumulative_prob + self.crop_replacement_prob):
                method = 'crop_replacement'
            elif random_value < (cumulative_prob := cumulative_prob + self.shuffle_prob):
                method = 'shuffle'
            elif random_value < (cumulative_prob := cumulative_prob + self.chunk_shuffle_prob):
                method = 'chunk_shuffle'
            else:
                method = 'unchanged'

            if method == 'reverse':
                attacked = torch.flip(watermarked, dims=[2])
                updated_original = torch.flip(updated_original, dims=[2])
                ground_truth_presence = torch.flip(ground_truth_presence, dims=[2])

                if num_samples > 1:
                    pass  # Reverse is inherently correct with torch.flip

                self.stats['reverse'] += batch_size
            elif method == 'head_trim':
                # Models real-world audio cropping from the beginning
                if num_samples > self.max_trim_samples:
                    trim_amount = torch.randint(1, self.max_trim_samples + 1, (1,)).item()
                else:
                    trim_amount = max(1, num_samples // 10)

                attacked = watermarked[:, :, trim_amount:]
                updated_original = updated_original[:, :, trim_amount:]
                ground_truth_presence = ground_truth_presence[:, :, trim_amount:]

                pad_width = trim_amount
                attacked = torch.nn.functional.pad(attacked, (0, pad_width))
                updated_original = torch.nn.functional.pad(updated_original, (0, pad_width))
                ground_truth_presence = torch.nn.functional.pad(ground_truth_presence, (0, pad_width))

                self.stats['head_trim'] += batch_size
            elif method == 'tail_trim':
                # Models real-world audio cropping from the end
                if num_samples > self.max_trim_samples:
                    trim_amount = torch.randint(1, self.max_trim_samples + 1, (1,)).item()
                else:
                    trim_amount = max(1, num_samples // 10)

                attacked = watermarked[:, :, :-trim_amount]
                updated_original = updated_original[:, :, :-trim_amount]
                ground_truth_presence = ground_truth_presence[:, :, :-trim_amount]

                pad_width = trim_amount
                attacked = torch.nn.functional.pad(attacked, (pad_width, 0))
                updated_original = torch.nn.functional.pad(updated_original, (pad_width, 0))
                ground_truth_presence = torch.nn.functional.pad(ground_truth_presence, (pad_width, 0))

                self.stats['tail_trim'] += batch_size
            elif method == 'crop_replacement':
                # True crop + pad, replacing the unrealistic circular shift
                if num_samples > self.max_trim_samples:
                    trim_amount = torch.randint(1, self.max_trim_samples + 1, (1,)).item()
                else:
                    trim_amount = max(1, num_samples // 10)

                # Head crop is the most common cropping scenario
                attacked = watermarked[:, :, trim_amount:]
                updated_original = updated_original[:, :, trim_amount:]
                ground_truth_presence = ground_truth_presence[:, :, trim_amount:]

                pad_width = trim_amount
                attacked = torch.nn.functional.pad(attacked, (0, pad_width))
                updated_original = torch.nn.functional.pad(updated_original, (0, pad_width))
                ground_truth_presence = torch.nn.functional.pad(ground_truth_presence, (0, pad_width))

                self.stats['crop_replacement'] += batch_size
            elif method == 'shuffle':
                segment_size = int(self.segment_duration * self.sample_rate)

                if segment_size <= 0:
                    logger.warning(f"Invalid segment_size={segment_size} (segment_duration={self.segment_duration}, sample_rate={self.sample_rate}). Skipping shuffle.")
                    attacked = watermarked
                    method = 'unchanged'
                    self.stats['unchanged'] += batch_size
                else:
                    num_segments = num_samples // segment_size
                    remainder = num_samples % segment_size

                    if num_segments >= 2:
                        complete_size = num_segments * segment_size

                        segments_watermarked = watermarked[:, :, :complete_size].unfold(2, segment_size, segment_size)
                        segments_original = updated_original[:, :, :complete_size].unfold(2, segment_size, segment_size)
                        segments_gt = ground_truth_presence[:, :, :complete_size].unfold(2, segment_size, segment_size)

                        shuffled_indices = torch.randperm(num_segments)

                        shuffled_watermarked = segments_watermarked[:, :, shuffled_indices].contiguous().view(batch_size, num_channels, -1)
                        shuffled_original = segments_original[:, :, shuffled_indices].contiguous().view(batch_size, num_channels, -1)
                        shuffled_gt = segments_gt[:, :, shuffled_indices].contiguous().view(batch_size, num_channels, -1)

                        if remainder > 0:
                            attacked = torch.cat([shuffled_watermarked, watermarked[:, :, complete_size:]], dim=2)
                            updated_original = torch.cat([shuffled_original, updated_original[:, :, complete_size:]], dim=2)
                            ground_truth_presence = torch.cat([shuffled_gt, ground_truth_presence[:, :, complete_size:]], dim=2)
                        else:
                            attacked = shuffled_watermarked
                            updated_original = shuffled_original
                            ground_truth_presence = shuffled_gt

                        self.stats['shuffle'] += batch_size
                    else:
                        attacked = watermarked
                        method = 'unchanged'
                        self.stats['unchanged'] += batch_size
            elif method == 'chunk_shuffle':
                chunk_size = num_samples // self.chunk_divisions

                if chunk_size > 0 and num_samples > 2 * chunk_size:
                    chunk1_start = torch.randint(0, num_samples - chunk_size, (1,)).item()

                    max_attempts = 100
                    attempts = 0
                    chunk2_start = torch.randint(0, num_samples - chunk_size, (1,)).item()

                    while abs(chunk1_start - chunk2_start) < chunk_size and attempts < max_attempts:
                        chunk2_start = torch.randint(0, num_samples - chunk_size, (1,)).item()
                        attempts += 1

                    if attempts < max_attempts:
                        indices = torch.arange(num_samples, device=watermarked.device)

                        chunk1_indices = indices[chunk1_start:chunk1_start+chunk_size].clone()
                        chunk2_indices = indices[chunk2_start:chunk2_start+chunk_size].clone()
                        indices[chunk1_start:chunk1_start+chunk_size] = chunk2_indices
                        indices[chunk2_start:chunk2_start+chunk_size] = chunk1_indices

                        attacked = watermarked[:, :, indices]
                        updated_original = updated_original[:, :, indices]
                        ground_truth_presence = ground_truth_presence[:, :, indices]

                        self.stats['chunk_shuffle'] += batch_size
                    else:
                        attacked = watermarked
                        method = 'unchanged'
                        self.stats['unchanged'] += batch_size
                else:
                    attacked = watermarked
                    method = 'unchanged'
                    self.stats['unchanged'] += batch_size
            else:
                attacked = watermarked
                self.stats['unchanged'] += batch_size

            assert attacked.shape == watermarked.shape, \
                f"Shape mismatch: attacked {attacked.shape} != watermarked {watermarked.shape}"
            assert updated_original.shape == watermarked.shape, \
                f"Shape mismatch: updated_original {updated_original.shape} != watermarked {watermarked.shape}"
            assert ground_truth_presence.shape == watermarked.shape, \
                f"Shape mismatch: ground_truth_presence {ground_truth_presence.shape} != watermarked {watermarked.shape}"

            total_samples = batch_size
            for key in self.stats:
                self.stats[key] = float((self.stats[key] / total_samples) * 100)

            return attacked, updated_original, ground_truth_presence, self.stats, method

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to apply attack: {str(e)}") from e

# --- Demo ---

def main() -> None:
    """
    Example test function for SequenceAttacks module.

    This function demonstrates the usage of SequenceAttacks by:
    1. Loading audio files from a test directory
    2. Creating a batch of audio samples
    3. Applying attacks to the batch
    4. Visualizing and saving the results

    Raises:
        FileNotFoundError: If audio_samples directory is not found.
        ValueError: If insufficient audio files are available.
        RuntimeError: If audio processing fails.
    """
    try:
        sample_rate = 16000
        duration = 3.0
        samples_to_keep = int(duration * sample_rate)

        logger.info(f"Initializing test with sample_rate={sample_rate}, duration={duration}s")

        seq_attacker = SequenceAttacks(
            sample_rate=sample_rate,
            reverse_prob=0.05,
            head_trim_prob=0.6,
            tail_trim_prob=0.25,
            crop_replacement_prob=0.0,
            shuffle_prob=0.1,
            chunk_shuffle_prob=0.0,
            segment_duration=0.05,
            chunk_divisions=4,
            max_trim_ms=12.0
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        audio_folder = os.path.join(parent_dir, "audio_samples")

        if not os.path.exists(audio_folder):
            raise FileNotFoundError(f"Audio samples directory not found: {audio_folder}")

        logger.info(f"Looking for audio files in: {audio_folder}")

        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3'))][:4]

        if len(audio_files) < 2:
            raise ValueError(f"Need at least 2 audio files in {audio_folder}, found {len(audio_files)}")

        logger.info(f"Found {len(audio_files)} audio files for testing")

        audio_tensors = []

        for audio_file in audio_files:
            try:
                file_path = os.path.join(audio_folder, audio_file)
                waveform, original_sample_rate = torchaudio.load(file_path)

                logger.debug(f"Loaded {audio_file}: shape={waveform.shape}, sr={original_sample_rate}")

                if original_sample_rate != sample_rate:
                    resampler = torchaudio.transforms.Resample(original_sample_rate, sample_rate)
                    waveform = resampler(waveform)
                    logger.debug(f"Resampled {audio_file} from {original_sample_rate}Hz to {sample_rate}Hz")

                if waveform.shape[1] > samples_to_keep:
                    waveform = waveform[:, :samples_to_keep]
                elif waveform.shape[1] < samples_to_keep:
                    padding = samples_to_keep - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                    logger.debug(f"Padded {audio_file} with {padding} samples")

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    logger.debug(f"Converted {audio_file} to mono")

                audio_tensors.append(waveform)

            except Exception as e:
                logger.error(f"Failed to load {audio_file}: {str(e)}", exc_info=True)
                raise

        original_batch = torch.stack(audio_tensors)
        logger.info(f"Created audio batch with shape: {original_batch.shape}")

        noise = torch.randn_like(original_batch) * NOISE_LEVEL
        watermarked_batch = original_batch + noise

        ground_truth_presence = torch.zeros_like(original_batch)

        block_size = samples_to_keep // 10
        for block_idx in range(0, samples_to_keep, block_size * 2):
            if block_idx + block_size <= samples_to_keep:
                ground_truth_presence[:, :, block_idx:block_idx+block_size] = 1.0

        logger.debug(f"Created ground truth presence with {ground_truth_presence.mean().item():.2%} watermark coverage")

        output_dir = "output_seq"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        attacked, updated_original, updated_presence, attack_stats, applied_method = seq_attacker(
            original_batch, watermarked_batch, ground_truth_presence
        )

        logger.info("\nSequence Attack Statistics:")
        for key, value in attack_stats.items():
            logger.info(f"  {key}: {value:.2f}%")
        logger.info(f"Attack Method Applied: {applied_method}")

        logger.info("Generating visualization plots...")

        for sample_idx, (original_sample, updated_original_sample, watermarked_sample,
                        attacked_sample, presence_sample) in enumerate(
                        zip(original_batch, updated_original, watermarked_batch,
                            attacked, updated_presence)):
            try:
                plt.figure(figsize=(15, 18))

                plt.subplot(5, 1, 1)
                plt.plot(original_sample.squeeze().numpy())
                plt.title(f'Original Audio {sample_idx+1}')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)

                plt.subplot(5, 1, 2)
                plt.plot(updated_original_sample.squeeze().numpy())
                plt.title(f'Updated Original Audio {sample_idx+1} (After {applied_method.title()} Attack)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)

                plt.subplot(5, 1, 3)
                plt.plot(watermarked_sample.squeeze().numpy())
                plt.title(f'Watermarked Audio {sample_idx+1} (Before Attack)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)

                ax = plt.subplot(5, 1, 4)
                attacked_data = attacked_sample.squeeze().numpy()
                plt.plot(attacked_data)
                plt.title(f'Attacked Audio {sample_idx+1} (After {applied_method.title()} Attack)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)

                y_min, y_max = plt.ylim()
                num_samples_plot = len(attacked_data)

                legend_elements = [
                    patches.Patch(facecolor='lightgreen', alpha=0.3, label='Reverse'),
                    patches.Patch(facecolor='lightblue', alpha=0.3, label='Circular Shift'),
                    patches.Patch(facecolor='salmon', alpha=0.3, label='Shuffle'),
                    patches.Patch(facecolor='lightyellow', alpha=0.3, label='Chunk Shuffle'),
                    patches.Patch(facecolor='white', label='Unchanged')
                ]

                method_color_map = {
                    'reverse': 'lightgreen',
                    'circular_shift': 'lightblue',
                    'shuffle': 'salmon',
                    'chunk_shuffle': 'lightyellow',
                    'unchanged': 'white'
                }

                background_color = method_color_map.get(applied_method, 'white')
                ax.add_patch(patches.Rectangle((0, y_min), num_samples_plot, y_max - y_min,
                                             facecolor=background_color, alpha=0.3))

                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

                plt.subplot(5, 1, 5)
                plt.plot(presence_sample.squeeze().numpy(), linewidth=2)
                plt.title('Ground Truth Watermark Presence (1: Present, 0: Absent)')
                plt.xlabel('Sample Number')
                plt.ylabel('Presence Indicator')
                plt.grid(True, alpha=0.3)
                plt.ylim(-0.1, 1.1)

                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Absent')
                plt.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='Present')
                plt.legend(loc='upper right', fontsize=10)

                plt.tight_layout(pad=2.0)

                plot_filename = os.path.join(output_dir, f'waveforms_{sample_idx+1}.png')
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()

                logger.debug(f"Saved visualization plot: {plot_filename}")

            except Exception as e:
                logger.error(f"Failed to generate plot for sample {sample_idx+1}: {str(e)}", exc_info=True)
                plt.close()

        logger.info("Saving audio files...")

        def prepare_audio_for_save(tensor: torch.Tensor) -> torch.Tensor:
            """
            Prepare audio tensor for saving to file.

            Args:
                tensor (torch.Tensor): Input audio tensor

            Returns:
                torch.Tensor: Properly formatted tensor [channels, samples]
            """
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.squeeze(0)

            tensor = tensor.float()

            if tensor.size(0) > 1:
                tensor = tensor.mean(dim=0, keepdim=True)

            return tensor

        for sample_idx, (original_sample, updated_original_sample, watermarked_sample,
                        attacked_sample) in enumerate(
                        zip(original_batch, updated_original, watermarked_batch,
                            attacked)):
            try:
                file_paths = [
                    (f"original_{sample_idx}.wav", original_sample),
                    (f"updated_original_{sample_idx}.wav", updated_original_sample),
                    (f"watermarked_{sample_idx}.wav", watermarked_sample),
                    (f"attacked_{sample_idx}.wav", attacked_sample)
                ]

                for filename, audio_tensor in file_paths:
                    file_path = os.path.join(output_dir, filename)
                    prepared_audio = prepare_audio_for_save(audio_tensor)

                    torchaudio.save(
                        file_path,
                        prepared_audio,
                        sample_rate,
                        encoding='PCM_S',
                        bits_per_sample=16
                    )

                    logger.debug(f"Saved audio file: {file_path}")

            except Exception as e:
                logger.error(f"Failed to save audio for sample {sample_idx}: {str(e)}", exc_info=True)
                raise

        logger.info(f"\nSuccessfully completed! All outputs saved in '{output_dir}'")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sequence_attacks.log', mode='a')
        ]
    )

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
