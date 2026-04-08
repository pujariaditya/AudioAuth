"""
Audio Attack Pipeline

This module implements a three-stage attack pipeline for watermarked audio:
1. Localization Attacks: Segment-level modifications (20% of segments)
2. Sequence Attacks: Batch-level transformations (reverse, shift, shuffle)
3. Effect Attacks: Audio effect transformations (noise, filters, volume changes)

The pipeline maintains ground truth alignment throughout all stages to ensure
accurate watermark presence tracking.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchaudio

from .localization_attacks import LocalizationAttacks, load_and_preprocess_audio
from .sequence_attacks import SequenceAttacks
from .effect_attacks import EffectAttacks

# These constants are only used in the main() demo function, not in the actual pipeline
DEMO_SAMPLE_RATE = 16000  # Hz (demo only)
DEMO_WINDOW_DURATION = 0.1  # seconds for localization segments (demo only)
DEMO_AUDIO_DURATION = 3.0  # seconds (demo only)
DEMO_NOISE_AMPLITUDE = 0.0001  # for watermarking simulation (demo only)

OUTPUT_BASE = "output_combined"
STAGE1_DIR = "stage1_localization"
STAGE2_DIR = "stage2_sequence"
VIZ_DIR = "visualizations"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class AttackPipeline(torch.nn.Module):
    """
    Implements a three-stage attack pipeline for watermarked audio.

    This pipeline applies localization attacks, followed by sequence attacks,
    and finally effect attacks, maintaining ground truth consistency throughout
    the process.

    Attributes:
        sample_rate: Audio sampling rate in Hz.
        window_duration: Duration of each localization segment window in seconds.
        localization_attack: LocalizationAttacks module (Stage 1).
        sequence_attack: SequenceAttacks module (Stage 2).
        effect_attack: EffectAttacks module (Stage 3).
        pipeline_stats: Dict accumulating statistics from all three stages.
    """

    def __init__(
        self,
        sample_rate: int,
        window_duration: float,
        # Localization attack parameters
        target_ratio: float,
        original_revert_prob: float,
        zero_replace_prob: float,
        # Sequence attack parameters
        reverse_prob: float,
        head_trim_prob: float,
        tail_trim_prob: float,
        crop_replacement_prob: float,
        shuffle_prob: float,
        chunk_shuffle_prob: float,
        segment_duration: float,
        chunk_divisions: int,
        max_trim_ms: float = 12.0,
        # Effect attack parameters
        effect_enabled: Dict[str, bool] = None,  # Required
        effect_params: Optional[Dict[str, Any]] = None,
        scheduler_config: Dict[str, Any] = None  # Required scheduler configuration
    ) -> None:
        """
        Initialize the attack pipeline.

        Args:
            sample_rate: Audio sampling rate in Hz
            window_duration: Window duration for localization segments
            target_ratio: Target ratio of segments to attack (0-1)
            original_revert_prob: Probability of reverting to original
            zero_replace_prob: Probability of replacing with zeros
            reverse_prob: Probability of reversing audio
            head_trim_prob: Probability of trimming samples from head (realistic cropping)
            tail_trim_prob: Probability of trimming samples from tail (realistic cropping)
            crop_replacement_prob: Probability of crop replacement (replaces circular shift)
            shuffle_prob: Probability of shuffling segments
            chunk_shuffle_prob: Probability of chunk shuffle
            segment_duration: Duration for shuffle segments (in seconds)
            chunk_divisions: Number of chunks for chunk shuffle
            max_trim_ms: Maximum trim duration in milliseconds (default: 12ms)
            effect_enabled: Dict of enabled flags for each effect (required)
            effect_params: Dict of parameters for each effect with choices format (optional)
            scheduler_config: Configuration for EffectScheduler (required)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.window_duration = window_duration

        self.localization_attack = LocalizationAttacks(
            sample_rate,
            window_duration,
            target_ratio=target_ratio,
            original_revert_prob=original_revert_prob,
            zero_replace_prob=zero_replace_prob
        )
        self.sequence_attack = SequenceAttacks(
            sample_rate,
            reverse_prob=reverse_prob,
            head_trim_prob=head_trim_prob,
            tail_trim_prob=tail_trim_prob,
            crop_replacement_prob=crop_replacement_prob,
            shuffle_prob=shuffle_prob,
            chunk_shuffle_prob=chunk_shuffle_prob,
            segment_duration=segment_duration,
            chunk_divisions=chunk_divisions,
            max_trim_ms=max_trim_ms
        )
        self.effect_attack = EffectAttacks(
            sample_rate=sample_rate,
            effect_enabled=effect_enabled,
            effect_params=effect_params,
            scheduler_config=scheduler_config
        )

        self.pipeline_stats = {
            'stage1_localization': {},
            'stage2_sequence': {},
            'stage3_effects': {},
            'combined': {}
        }


    def update_scheduler_metrics(
        self,
        effect_name: str,
        effect_params: Dict[str, Any],
        ber: float,
        miou: float
    ) -> None:
        """
        Update scheduler metrics for a specific effect and parameter combination.

        Args:
            effect_name: Name of the effect
            effect_params: Parameters used for this effect
            ber: Localized bit error rate measurement (computed only in watermarked regions)
            miou: Mean IoU measurement
        """
        if hasattr(self.effect_attack, 'scheduler') and self.effect_attack.scheduler is not None:
            self.effect_attack.scheduler.update_effect_metrics(
                effect_name=effect_name,
                effect_params=effect_params,
                localized_ber=ber,
                miou=miou
            )
            logger.debug(f"Updated scheduler metrics for {effect_name}: BER={ber:.4f}, mIoU={miou:.4f}")
        else:
            logger.debug("Scheduler not active, skipping metric update")

    def adapt_scheduler_probabilities(self) -> None:
        """
        Adapt scheduler effect probabilities based on accumulated metrics.
        """
        if hasattr(self.effect_attack, 'scheduler') and self.effect_attack.scheduler is not None:
            self.effect_attack.scheduler.adapt_effect_probabilities()
            logger.info("Adapted scheduler effect probabilities")

    def get_scheduler_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get scheduler statistics if available.

        Returns:
            Dictionary of scheduler statistics or None if scheduler not active
        """
        if hasattr(self.effect_attack, 'scheduler') and self.effect_attack.scheduler is not None:
            stats = self.effect_attack.scheduler.get_effect_statistics()

            if 'bandpass' in stats and hasattr(self.effect_attack.scheduler, 'parameter_success_rates'):
                bandpass_params = self.effect_attack.scheduler.parameter_success_rates.get('bandpass', {})

                freq_pair_stats = {}
                for param_key, success_list in bandpass_params.items():
                    if param_key[0] == 'frequency_pairs' and success_list:
                        freq_pair = param_key[1] if isinstance(param_key[1], tuple) else param_key[1]
                        success_rate = sum(success_list) / len(success_list)
                        freq_pair_stats[str(freq_pair)] = {
                            'success_rate': success_rate,
                            'samples': len(success_list)
                        }

                if freq_pair_stats:
                    stats['bandpass']['frequency_pair_stats'] = freq_pair_stats

            return stats
        return None

    def train(self, mode: bool = True):
        """
        Set the module in training mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)

        Returns:
            self
        """
        super().train(mode)
        self.localization_attack.train(mode)
        self.sequence_attack.train(mode) if hasattr(self.sequence_attack, 'train') else None
        self.effect_attack.train(mode) if hasattr(self.effect_attack, 'train') else None
        return self

    def eval(self):
        """
        Set the module in evaluation mode.

        Returns:
            self
        """
        return self.train(False)

    def process(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process audio through all three attack stages.

        Args:
            original: Original audio batch tensor [batch_size, channels, samples]
            watermarked: Watermarked audio batch tensor [batch_size, channels, samples]

        Returns:
            Tuple containing:
                - final_attacked: Final attacked audio tensor
                - final_ground_truth: Final ground truth presence tensor
                - final_updated_original: Final updated original tensor
                - pipeline_stats: Combined statistics from all stages
        """
        batch_size, num_channels, num_samples = watermarked.shape

        # --- Stage 1: Localization ---

        stage1_attacked, stage1_ground_truth, stage1_updated_original, stage1_stats = \
            self.localization_attack(original, watermarked)

        self.pipeline_stats['stage1_localization'] = stage1_stats

        # Channel 0 is the canonical mask channel; all channels carry the same
        # presence information so averaging over channel 0 is sufficient.
        watermark_coverage = stage1_ground_truth[:, 0, :].mean().item() * 100

        # --- Stage 2: Sequence Attacks ---

        # Stage 2 returns 5 values: attacked audio, updated original, ground truth,
        # stats dict, and applied method name.
        final_attacked, stage2_updated_original, stage2_ground_truth, stage2_stats, applied_method = \
            self.sequence_attack(
                stage1_updated_original,
                stage1_attacked,
                stage1_ground_truth
            )

        self.pipeline_stats['stage2_sequence'] = stage2_stats
        self.pipeline_stats['stage2_sequence']['method'] = applied_method

        stage2_watermark_coverage = stage2_ground_truth[:, 0, :].mean().item() * 100

        # --- Stage 3: Effect Attacks ---

        # Stage 3 returns only 2 tensors + stats (no updated original) because
        # effect attacks modify audio in-place without needing the original signal.
        final_effect_attacked, final_effect_ground_truth, effect_stats = \
            self.effect_attack(
                final_attacked,
                stage2_ground_truth
            )

        self.pipeline_stats['stage3_effects'] = effect_stats

        final_watermark_coverage = final_effect_ground_truth[:, 0, :].mean().item() * 100

        self.pipeline_stats['combined'] = {
            'initial_watermark_coverage': 100.0,
            'stage1_watermark_coverage': watermark_coverage,
            'stage2_watermark_coverage': stage2_watermark_coverage,
            'final_watermark_coverage': final_watermark_coverage,
            'watermark_reduction': 100.0 - final_watermark_coverage,
            'stage2_method': applied_method,
            'stage3_effects': effect_stats.get('effects_applied', [])
        }

        if 'effect_params_used' in effect_stats:
            self.pipeline_stats['effect_params_used'] = effect_stats['effect_params_used']

        return final_effect_attacked, final_effect_ground_truth, stage2_updated_original, self.pipeline_stats

# --- Visualization ---

def create_pipeline_visualization(
    original: torch.Tensor,
    watermarked: torch.Tensor,
    stage1_attacked: torch.Tensor,
    stage1_ground_truth: torch.Tensor,
    stage2_attacked: torch.Tensor,
    stage2_ground_truth: torch.Tensor,
    final_attacked: torch.Tensor,
    final_ground_truth: torch.Tensor,
    pipeline_stats: Dict[str, Any],
    sample_idx: int,
    output_dir: Path
) -> None:
    """
    Create comprehensive visualization of the pipeline stages.

    Args:
        original: Original audio tensor
        watermarked: Initial watermarked audio
        stage1_attacked: Audio after stage 1
        stage1_ground_truth: Ground truth after stage 1
        stage2_attacked: Audio after stage 2
        stage2_ground_truth: Ground truth after stage 2
        final_attacked: Final attacked audio (after stage 3)
        final_ground_truth: Final ground truth (after stage 3)
        pipeline_stats: Pipeline statistics
        sample_idx: Sample index for labeling
        output_dir: Output directory for saving plots
    """
    fig, axes = plt.subplots(8, 1, figsize=(16, 26))

    original_data = original.squeeze().cpu().numpy()
    watermarked_data = watermarked.squeeze().cpu().numpy()
    stage1_data = stage1_attacked.squeeze().cpu().numpy()
    stage1_gt_data = stage1_ground_truth.squeeze().cpu().numpy()
    stage2_data = stage2_attacked.squeeze().cpu().numpy()
    stage2_gt_data = stage2_ground_truth.squeeze().cpu().numpy()
    final_data = final_attacked.squeeze().cpu().numpy()
    final_gt_data = final_ground_truth.squeeze().cpu().numpy()

    axes[0].plot(original_data, color='blue', alpha=0.7)
    axes[0].set_title(f'Original Audio (Sample {sample_idx+1})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(watermarked_data, color='green', alpha=0.7)
    axes[1].set_title('Initial Watermarked Audio (100% watermark coverage)', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(stage1_data, color='orange', alpha=0.7)
    stage1_coverage = pipeline_stats['combined']['stage1_watermark_coverage']
    stage1_avg = stage1_coverage['average'] if isinstance(stage1_coverage, dict) else stage1_coverage
    axes[2].set_title(f'After Stage 1: Localization Attacks ({stage1_avg:.1f}% watermark coverage)',
                     fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    axes[3].fill_between(range(len(stage1_gt_data)), 0, stage1_gt_data,
                         color='orange', alpha=0.5)
    axes[3].set_title('Stage 1 Ground Truth (1=watermark present, 0=absent)', fontsize=12)
    axes[3].set_ylabel('Presence')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(stage2_data, color='purple', alpha=0.7)
    method = pipeline_stats['stage2_sequence'].get('method', 'unknown')
    stage2_coverage = pipeline_stats['combined'].get('stage2_watermark_coverage', 0)
    stage2_avg = stage2_coverage['average'] if isinstance(stage2_coverage, dict) else stage2_coverage
    axes[4].set_title(f'After Stage 2: Sequence Attacks ({method}) - {stage2_avg:.1f}% watermark coverage',
                     fontsize=12, fontweight='bold')
    axes[4].set_ylabel('Amplitude')
    axes[4].grid(True, alpha=0.3)

    axes[5].fill_between(range(len(stage2_gt_data)), 0, stage2_gt_data,
                         color='purple', alpha=0.5)
    axes[5].set_title('Stage 2 Ground Truth (1=watermark present, 0=absent)', fontsize=12)
    axes[5].set_ylabel('Presence')
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(final_data, color='red', alpha=0.7)
    effects = pipeline_stats['combined'].get('stage3_effects', [])
    effects_str = ', '.join(effects[:3]) if effects else 'various'
    if len(effects) > 3:
        effects_str += '...'
    final_coverage = pipeline_stats['combined']['final_watermark_coverage']
    final_avg = final_coverage['average'] if isinstance(final_coverage, dict) else final_coverage
    axes[6].set_title(f'After Stage 3: Effect Attacks ({effects_str}) - {final_avg:.1f}% watermark coverage',
                     fontsize=12, fontweight='bold')
    axes[6].set_ylabel('Amplitude')
    axes[6].grid(True, alpha=0.3)

    axes[7].fill_between(range(len(final_gt_data)), 0, final_gt_data,
                         color='red', alpha=0.5)
    axes[7].set_title('Final Ground Truth (after all three stages)', fontsize=12)
    axes[7].set_xlabel('Sample Number')
    axes[7].set_ylabel('Presence')
    axes[7].set_ylim(-0.1, 1.1)
    axes[7].grid(True, alpha=0.3)

    fig.text(0.02, 0.87, 'INPUT', fontsize=14, fontweight='bold',
            rotation=90, va='center', color='darkgreen')
    fig.text(0.02, 0.62, 'STAGE 1', fontsize=14, fontweight='bold',
            rotation=90, va='center', color='darkorange')
    fig.text(0.02, 0.40, 'STAGE 2', fontsize=14, fontweight='bold',
            rotation=90, va='center', color='purple')
    fig.text(0.02, 0.17, 'STAGE 3', fontsize=14, fontweight='bold',
            rotation=90, va='center', color='darkred')

    plt.tight_layout(rect=[0.03, 0, 1, 1])

    plot_path = output_dir / f'pipeline_visualization_sample_{sample_idx+1}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved pipeline visualization: {plot_path}")

# --- Utility Functions ---

def save_audio_outputs(
    audio_dict: Dict[str, torch.Tensor],
    sample_rate: int,
    output_dir: Path
) -> None:
    """
    Save audio tensors to WAV files.

    Args:
        audio_dict: Dictionary of audio tensors to save
        sample_rate: Sample rate for saving
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, audio_tensor in audio_dict.items():
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        elif audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_tensor = audio_tensor.float().cpu()

        file_path = output_dir / f"{name}.wav"
        torchaudio.save(
            file_path,
            audio_tensor,
            sample_rate,
            encoding='PCM_S',
            bits_per_sample=16
        )
        logger.debug(f"Saved audio: {file_path}")

def save_pipeline_report(
    pipeline_stats: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Save comprehensive pipeline report.

    Args:
        pipeline_stats: Pipeline statistics
        output_dir: Output directory
    """
    report_path = output_dir / "pipeline_report.txt"

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CASCADED ATTACK PIPELINE REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

        f.write("STAGE 1: LOCALIZATION ATTACKS\n")
        f.write("-"*40 + "\n")
        for key, value in pipeline_stats['stage1_localization'].items():
            f.write(f"  {key}: {value:.2f}%\n")
        f.write("\n")

        f.write("STAGE 2: SEQUENCE ATTACKS\n")
        f.write("-"*40 + "\n")
        f.write(f"  Method Applied: {pipeline_stats['stage2_sequence'].get('method', 'N/A')}\n")
        for key, value in pipeline_stats['stage2_sequence'].items():
            if key != 'method':
                f.write(f"  {key}: {value:.2f}%\n")
        f.write("\n")

        f.write("STAGE 3: EFFECT ATTACKS\n")
        f.write("-"*40 + "\n")
        effects = pipeline_stats.get('stage3_effects', {}).get('effects_applied', [])
        f.write(f"  Effects Applied: {', '.join(effects) if effects else 'N/A'}\n")
        effect_stats = pipeline_stats.get('stage3_effects', {}).get('effect_stats', {})
        if effect_stats:
            for key, value in effect_stats.items():
                if key != 'total':
                    f.write(f"    {key}: {value} times\n")
        f.write("\n")

        f.write("COMBINED PIPELINE RESULTS\n")
        f.write("-"*40 + "\n")
        for key, value in pipeline_stats['combined'].items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.2f}%\n")
            elif isinstance(value, list):
                f.write(f"  {key}: {', '.join(value) if value else 'N/A'}\n")
            else:
                f.write(f"  {key}: {value}\n")

    json_path = output_dir / "pipeline_stats.json"
    with open(json_path, 'w') as f:
        json.dump(pipeline_stats, f, indent=2)

    logger.info(f"Saved pipeline report: {report_path}")
    logger.info(f"Saved pipeline stats: {json_path}")

# --- Demo ---

def main() -> None:
    """
    Main function to demonstrate the attack pipeline.

    This function:
    1. Loads audio samples from the audio_samples directory
    2. Creates synthetic watermarked versions
    3. Applies the attack pipeline
    4. Saves outputs and visualizations
    """
    try:
        logger.info("\n" + "="*60)
        logger.info("CASCADED ATTACK PIPELINE DEMONSTRATION")
        logger.info("="*60)

        pipeline = AttackPipeline(
            sample_rate=DEMO_SAMPLE_RATE,
            window_duration=DEMO_WINDOW_DURATION,
            target_ratio=0.20,
            original_revert_prob=0.50,
            zero_replace_prob=0.50,
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

        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        audio_folder = parent_dir / "audio_samples"

        if not audio_folder.exists():
            raise FileNotFoundError(f"Audio samples directory not found: {audio_folder}")

        logger.info(f"Loading audio files from: {audio_folder}")

        audio_files = list(audio_folder.glob("*.wav")) + list(audio_folder.glob("*.mp3"))
        audio_files = audio_files[:4]

        if len(audio_files) < 1:
            raise ValueError(f"Need at least 1 audio file, found {len(audio_files)}")

        logger.info(f"Found {len(audio_files)} audio files")

        audio_tensors = []
        for audio_file in audio_files:
            waveform = load_and_preprocess_audio(
                str(audio_file),
                DEMO_SAMPLE_RATE,
                DEMO_AUDIO_DURATION
            )
            audio_tensors.append(waveform)

        original_batch = torch.stack(audio_tensors)
        logger.info(f"Created batch with shape: {original_batch.shape}")

        noise = torch.randn_like(original_batch) * DEMO_NOISE_AMPLITUDE
        watermarked_batch = original_batch + noise
        logger.info("Created synthetic watermarked audio")

        output_base = Path(OUTPUT_BASE)
        output_base.mkdir(exist_ok=True)

        stage1_dir = output_base / STAGE1_DIR
        stage2_dir = output_base / STAGE2_DIR
        stage3_dir = output_base / "stage3_effects"
        viz_dir = output_base / VIZ_DIR

        for dir_path in [stage1_dir, stage2_dir, stage3_dir, viz_dir]:
            dir_path.mkdir(exist_ok=True)

        final_attacked, final_ground_truth, final_updated_original, pipeline_stats = \
            pipeline.process(original_batch, watermarked_batch)

        logger.info("\n" + "="*60)
        logger.info("SAVING OUTPUTS")
        logger.info("="*60)

        stage1_attacked, stage1_ground_truth, stage1_updated_original, _ = \
            pipeline.localization_attack(original_batch, watermarked_batch)

        # Stage 2 skipped in demo; reuse stage 1 outputs
        stage2_attacked = stage1_attacked
        stage2_updated_original = stage1_updated_original
        stage2_ground_truth = stage1_ground_truth

        for idx in range(len(audio_tensors)):
            save_audio_outputs(
                {
                    f"sample_{idx+1}_stage1_attacked": stage1_attacked[idx],
                    f"sample_{idx+1}_stage1_original": stage1_updated_original[idx]
                },
                DEMO_SAMPLE_RATE,
                stage1_dir
            )

            save_audio_outputs(
                {
                    f"sample_{idx+1}_original": original_batch[idx],
                    f"sample_{idx+1}_initial_watermarked": watermarked_batch[idx],
                    f"sample_{idx+1}_final_attacked": final_attacked[idx],
                    f"sample_{idx+1}_final_original": final_updated_original[idx]
                },
                DEMO_SAMPLE_RATE,
                stage3_dir
            )

            create_pipeline_visualization(
                original_batch[idx],
                watermarked_batch[idx],
                stage1_attacked[idx],
                stage1_ground_truth[idx],
                stage2_attacked[idx],
                stage2_ground_truth[idx],
                final_attacked[idx],
                final_ground_truth[idx],
                pipeline_stats,
                idx,
                viz_dir
            )

        save_pipeline_report(pipeline_stats, output_base)

        torch.save({
            'stage1_ground_truth': stage1_ground_truth,
            'stage2_ground_truth': stage2_ground_truth,
            'final_ground_truth': final_ground_truth
        }, output_base / 'ground_truth_tensors.pt')

        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*60)
        logger.info(f"All outputs saved to: {output_base}")
        logger.info("\nSummary:")
        logger.info(f"  - Stage 1 outputs: {stage1_dir}")
        logger.info(f"  - Stage 2 outputs: {stage2_dir}")
        logger.info(f"  - Stage 3 outputs: {stage3_dir}")
        logger.info(f"  - Visualizations: {viz_dir}")
        logger.info(f"  - Report: {output_base / 'pipeline_report.txt'}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('attack_pipeline.log', mode='a')
        ]
    )

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)
