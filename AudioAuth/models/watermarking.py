"""
Audio watermarking module combining generator and detector models.

Provides end-to-end watermarking functionality including watermark
embedding and detection for audio signals.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attacks.main import AttackPipeline

logger = logging.getLogger(__name__)


class AudioWatermarking(nn.Module):
    """Audio watermarking system with generator and detector.

    Combines watermark generation and detection into a unified system
    that operates in different phases (training, validation, inference).
    Always applies attacks between generation and detection to improve
    robustness during training.

    Attributes:
        generator (nn.Module): Model for generating watermark embeddings.
        detector (nn.Module): Model for detecting watermarks in audio.
        locator (nn.Module): Model for localizing watermarks temporally.
        sample_rate (int): Sample rate of audio in Hz.
        localization_threshold (float): Threshold for watermark presence
            detection, sourced from locator config or defaulting to 0.5.
        train_phase (str): Identifier selecting training-mode forward logic.
        valid_phase (str): Identifier selecting validation-mode forward logic.
        audio_sample_phase (str): Identifier selecting audio-sample-mode
            forward logic.
        train_attacks_config (Dict[str, Any]): Raw attack config dict for
            training phase.
        valid_attacks_config (Dict[str, Any]): Raw attack config dict for
            validation phase.
        train_attacks_pipeline (AttackPipeline | None): Instantiated training
            attack pipeline.
        valid_attacks_pipeline (AttackPipeline | None): Instantiated validation
            attack pipeline.
        attack_stats (dict): Most recent attack statistics for monitoring.
    """

    def __init__(
        self,
        generator: nn.Module,
        detector: nn.Module,
        locator: nn.Module,
        sample_rate: int,
        train_phase: str,
        valid_phase: str,
        audio_sample_phase: str,
        train_attacks_config: Dict[str, Any],
        valid_attacks_config: Dict[str, Any]
    ) -> None:
        """Initialize the audio watermarking system.

        Args:
            generator: Neural network model for watermark generation.
            detector: Neural network model for watermark detection.
            locator: Neural network model for watermark localization.
            sample_rate: Audio sample rate in Hz.
            train_phase: Identifier for training phase.
            valid_phase: Identifier for validation phase.
            audio_sample_phase: Identifier for audio sample phase.
            train_attacks_config: Training-specific attacks config.
            valid_attacks_config: Validation-specific attacks config.

        Raises:
            TypeError: If generator, detector, or locator is not a nn.Module.
            ValueError: If sample_rate is not positive or phase identifiers
                are empty.
        """
        super().__init__()

        if not isinstance(generator, nn.Module):
            raise TypeError(f"Generator must be nn.Module, got {type(generator)}")
        if not isinstance(detector, nn.Module):
            raise TypeError(f"Detector must be nn.Module, got {type(detector)}")
        if not isinstance(locator, nn.Module):
            raise TypeError(f"Locator must be nn.Module, got {type(locator)}")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

        if not train_phase or not isinstance(train_phase, str):
            raise ValueError("train_phase must be a non-empty string")
        if not valid_phase or not isinstance(valid_phase, str):
            raise ValueError("valid_phase must be a non-empty string")
        if not audio_sample_phase or not isinstance(audio_sample_phase, str):
            raise ValueError("audio_sample_phase must be a non-empty string")

        # Catch mismatched sample rates early to avoid silent failures
        if hasattr(generator, 'sample_rate') and generator.sample_rate != sample_rate:
            raise ValueError(
                f"Generator sample_rate ({generator.sample_rate}Hz) doesn't match "
                f"AudioWatermarking sample_rate ({sample_rate}Hz)"
            )
        if hasattr(detector, 'sample_rate') and detector.sample_rate != sample_rate:
            raise ValueError(
                f"Detector sample_rate ({detector.sample_rate}Hz) doesn't match "
                f"AudioWatermarking sample_rate ({sample_rate}Hz)"
            )
        if hasattr(locator, 'sample_rate') and locator.sample_rate != sample_rate:
            raise ValueError(
                f"Locator sample_rate ({locator.sample_rate}Hz) doesn't match "
                f"AudioWatermarking sample_rate ({sample_rate}Hz)"
            )

        self.generator = generator
        self.detector = detector
        self.locator = locator
        self.sample_rate = sample_rate

        if hasattr(locator, 'config') and hasattr(locator.config, 'localization_threshold'):
            self.localization_threshold = locator.config.localization_threshold
        else:
            self.localization_threshold = 0.5
            logger.warning(f"Using default localization threshold: {self.localization_threshold}")

        self.train_phase = train_phase
        self.valid_phase = valid_phase
        self.audio_sample_phase = audio_sample_phase

        self.train_attacks_config = train_attacks_config
        self.valid_attacks_config = valid_attacks_config

        self.train_attacks_pipeline = self._create_attack_pipeline(
            self.train_attacks_config, sample_rate, "training"
        )
        self.valid_attacks_pipeline = self._create_attack_pipeline(
            self.valid_attacks_config, sample_rate, "validation"
        )

        logger.info("Initialized phase-specific attack pipelines")

        self.attack_stats = {}


    def _create_attack_pipeline(
        self,
        config: Dict[str, Any],
        sample_rate: int,
        phase_name: str,
        specific_effect: str = None,
        specific_params: Dict[str, Any] = None
    ) -> Union[AttackPipeline, None]:
        """Create an attack pipeline from configuration.

        Args:
            config: Attacks configuration dictionary.
            sample_rate: Audio sample rate in Hz.
            phase_name: Name of the phase for logging.
            specific_effect: If set, only this effect is enabled
                (for per-effect evaluation).
            specific_params: Parameters to use for ``specific_effect``
                (for systematic parameter testing).

        Returns:
            AttackPipeline if enabled, None otherwise.
        """
        if config is None:
            logger.info(f"No attack pipeline for {phase_name} phase")
            return None

        if 'enabled' not in config:
            raise ValueError(f"{phase_name} attacks_config.enabled is required")

        if not config['enabled']:
            logger.info(f"Attacks disabled for {phase_name} phase")
            return None

        if 'localization' not in config:
            raise ValueError(f"{phase_name} attacks_config.localization is required")
        if 'sequence' not in config:
            raise ValueError(f"{phase_name} attacks_config.sequence is required")
        if 'effect' not in config:
            raise ValueError(f"{phase_name} attacks_config.effect is required")

        loc_config = config['localization']
        seq_config = config['sequence']
        effect_config = config['effect']

        required_loc_fields = ['window_duration', 'target_ratio', 'original_revert_prob',
                               'zero_replace_prob']
        for field in required_loc_fields:
            if field not in loc_config:
                raise ValueError(f"{phase_name} attacks_config.localization.{field} is required")

        required_seq_fields = ['reverse_prob', 'crop_replacement_prob', 'shuffle_prob',
                               'chunk_shuffle_prob', 'segment_duration', 'chunk_divisions']
        for field in required_seq_fields:
            if field not in seq_config:
                raise ValueError(f"{phase_name} attacks_config.sequence.{field} is required")

        effect_enabled = effect_config.get('effect_enabled', None)

        if specific_effect:
            # Isolates a single effect for systematic robustness evaluation
            temp_enabled = {name: False for name in effect_enabled.keys()}
            temp_enabled[specific_effect] = True
            effect_enabled = temp_enabled

        effect_params = effect_config.get('effect_params', None)

        scheduler_config = effect_config.get('scheduler_config', None)

        # Pydantic models need conversion to plain dicts for the pipeline
        if effect_params and hasattr(effect_params, 'model_dump'):
            effect_params_dict = {}
            all_effects = ['white_noise', 'pink_noise', 'lowpass', 'highpass', 'bandpass', 'volume', 'identity',
                          'updown_resample', 'echo', 'boost_audio', 'duck_audio', 'speed',
                          'random_noise', 'smooth', 'mp3_compression', 'aac_compression', 'encodec_compression']
            for effect_name in all_effects:
                if hasattr(effect_params, effect_name):
                    effect_obj = getattr(effect_params, effect_name)
                    if effect_obj and hasattr(effect_obj, 'model_dump'):
                        effect_params_dict[effect_name] = effect_obj.model_dump()
                    elif effect_obj and isinstance(effect_obj, dict):
                        effect_params_dict[effect_name] = effect_obj
                    elif effect_obj:
                        effect_params_dict[effect_name] = effect_obj
            effect_params = effect_params_dict
        elif effect_params and isinstance(effect_params, dict):
            effect_params_dict = {}
            for effect_name, params in effect_params.items():
                if hasattr(params, 'model_dump'):
                    effect_params_dict[effect_name] = params.model_dump()
                elif isinstance(params, dict):
                    effect_params_dict[effect_name] = params
            effect_params = effect_params_dict

        if scheduler_config and hasattr(scheduler_config, 'model_dump'):
            scheduler_config = scheduler_config.model_dump()

        # Propagate compound chain settings into the scheduler so it can
        # decide when to apply multi-effect chains
        if scheduler_config is not None:
            scheduler_config['compound_chain_prob'] = effect_config.get('compound_chain_prob', 0.0)
            scheduler_config['max_chain_length'] = effect_config.get('max_chain_length', 3)

        # Isolates a single effect for systematic robustness evaluation:
        # overrides both the enabled map and params so the pipeline only
        # exercises the target effect with the exact requested parameters.
        if specific_params is not None and specific_effect:
            specific_effect_enabled = {}
            for effect_name in effect_enabled:
                specific_effect_enabled[effect_name] = (effect_name == specific_effect)

            effect_params = {specific_effect: specific_params}
            effect_enabled = specific_effect_enabled

        pipeline = AttackPipeline(
            sample_rate=sample_rate,
            window_duration=loc_config['window_duration'],
            target_ratio=loc_config['target_ratio'],
            original_revert_prob=loc_config['original_revert_prob'],
            zero_replace_prob=loc_config['zero_replace_prob'],
            reverse_prob=seq_config['reverse_prob'],
            head_trim_prob=seq_config['head_trim_prob'],
            tail_trim_prob=seq_config['tail_trim_prob'],
            crop_replacement_prob=seq_config['crop_replacement_prob'],
            shuffle_prob=seq_config['shuffle_prob'],
            chunk_shuffle_prob=seq_config['chunk_shuffle_prob'],
            segment_duration=seq_config['segment_duration'],
            chunk_divisions=seq_config['chunk_divisions'],
            max_trim_ms=seq_config['max_trim_ms'],
            effect_enabled=effect_enabled,
            effect_params=effect_params,
            scheduler_config=scheduler_config
        )

        return pipeline

    # --- Forward Pass ---

    def forward(
        self,
        signal: torch.Tensor,
        msg: torch.Tensor,
        phase: str = None,
        specific_effect: str = None,
        specific_params: Dict[str, Any] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Execute forward pass of the watermarking system.

        Processes input audio through watermark generation and detection.
        Behavior varies based on the specified phase.

        Args:
            signal: Input audio tensor [B, 1, T].
            msg: Watermark message tensor [B, nbits].
            phase: Operation phase -- must match one of ``train_phase``,
                ``valid_phase``, or ``audio_sample_phase``. Defaults to
                ``train_phase`` when None.
            specific_effect: Single effect to apply during validation
                (for per-effect evaluation).
            specific_params: Parameters for ``specific_effect``
                (for systematic parameter testing).

        Returns:
            Training phase: tuple of (reconstructed_signal,
                watermarked_signal, watermarked_detector_output,
                locator_output, mask, clean_detector_output,
                clean_locator_output).
            Validation/eval phase: dict with keys
                ``reconstructed_signal``, ``watermarked_signal``,
                ``watermarked_detector_output``,
                ``watermarked_locator_output``, ``results_dict``, etc.

        Raises:
            ValueError: If phase is not recognized.
            RuntimeError: If forward pass computation fails.
        """
        try:
            if signal.dim() != 3:
                raise ValueError(f"Signal must be 3D tensor [B, 1, T], got shape {signal.shape}")
            if msg.dim() != 2:
                raise ValueError(f"Message must be 2D tensor [B, nbits], got shape {msg.shape}")

            if phase is None:
                phase = self.train_phase

            logger.debug(f"Generating watermark for batch size {signal.shape[0]}")
            reconstructed_signal = self.generator(signal, msg)
            watermarked_signal = reconstructed_signal + signal

            if phase == self.train_phase:
                return self._forward_train(signal, reconstructed_signal, watermarked_signal)

            elif phase in [self.valid_phase, self.audio_sample_phase]:
                return self._forward_eval(signal, reconstructed_signal, watermarked_signal, phase, specific_effect, specific_params)

            else:
                raise ValueError(f"Unknown phase: {phase}. Expected one of {self.train_phase}, "
                               f"{self.valid_phase}, {self.audio_sample_phase}")

        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"AudioWatermarking forward pass failed: {str(e)}") from e

    def _forward_train(
        self,
        signal: torch.Tensor,
        reconstructed_signal: torch.Tensor,
        watermarked_signal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute training phase forward pass.

        Always applies attacks to simulate real-world conditions and
        improve detector robustness.

        Args:
            signal: Original input signal [B, 1, T].
            reconstructed_signal: Generated watermark embedding [B, 1, T].
            watermarked_signal: Signal with watermark [B, 1, T].

        Returns:
            Tuple of (reconstructed_signal, attacked_signal,
                watermarked_detector_output, locator_output, mask,
                clean_detector_output, clean_locator_output).
        """
        if self.train_attacks_pipeline is not None:
            logger.debug("Applying attacks in training")
            attacked_signal, ground_truth_mask, _, attack_stats = self.train_attacks_pipeline.process(
                signal,
                watermarked_signal
            )

            # Run same attacks on unwatermarked audio for negative-sample training
            clean_attacked, _, _, _ = self.train_attacks_pipeline.process(
                signal,
                signal
            )
        else:
            logger.debug("Training attacks disabled - using watermarked signal directly")
            attacked_signal = watermarked_signal
            clean_attacked = signal
            ground_truth_mask = torch.ones(watermarked_signal.size(0), 1, watermarked_signal.size(2),
                                          device=watermarked_signal.device)
            attack_stats = {'attacks': 'disabled'}

        self.attack_stats = attack_stats

        # Ground truth mask indicates where watermark remains after attacks
        mask = ground_truth_mask  # [B, 1, T]

        coverage = mask[:, 0, :].mean().item() * 100
        logger.debug(f"Attacks applied. Watermark coverage: {coverage:.1f}%")

        # Pipeline: Generator -> Attack -> Locator & Detector (independent)

        locator_output = self.locator(attacked_signal)
        clean_locator_output = self.locator(clean_attacked)

        watermarked_detector_output = self.detector(attacked_signal)
        clean_detector_output = self.detector(clean_attacked)


        logger.debug(f"Training forward: locator_output shape {locator_output.shape}, "
                    f"watermarked_detector_output shape {watermarked_detector_output.shape}, "
                    f"clean_detector_output shape {clean_detector_output.shape}, "
                    f"clean_locator_output shape {clean_locator_output.shape}, mask shape {mask.shape}")

        return reconstructed_signal, attacked_signal, watermarked_detector_output, locator_output, mask, clean_detector_output, clean_locator_output

    def _forward_eval(
        self,
        original_signal: torch.Tensor,
        reconstructed_signal: torch.Tensor,
        watermarked_signal: torch.Tensor,
        phase: str,
        specific_effect: str = None,
        specific_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute evaluation phase forward pass.

        Applies attacks during validation to test robustness.

        Args:
            original_signal: Original input signal [B, 1, T].
            reconstructed_signal: Generated watermark embedding [B, 1, T].
            watermarked_signal: Signal with watermark [B, 1, T].
            phase: Current evaluation phase.
            specific_effect: Single effect to apply (for per-effect evaluation).
            specific_params: Parameters for ``specific_effect``.

        Returns:
            Dictionary containing output tensors and metrics.
        """
        if phase == self.valid_phase and (self.valid_attacks_pipeline is not None or specific_effect):
            logger.debug("Applying attacks in validation for robustness testing")

            if specific_effect:
                specific_pipeline = self._create_attack_pipeline(
                    self.valid_attacks_config,
                    self.sample_rate,
                    f"validation_{specific_effect}",
                    specific_effect,
                    specific_params
                )
                if specific_pipeline is None:
                    raise ValueError(f"Failed to create pipeline for effect: {specific_effect}")

                attacked_signal, ground_truth_mask, _, attack_stats = specific_pipeline.process(
                    original_signal,
                    watermarked_signal
                )
            else:
                attacked_signal, ground_truth_mask, _, attack_stats = self.valid_attacks_pipeline.process(
                    original_signal,
                    watermarked_signal
                )

            detector_input = attacked_signal

            # Apply same attacks to clean signal for negative samples
            if specific_effect and specific_pipeline:
                clean_attacked, _, _, _ = specific_pipeline.process(
                    original_signal,
                    original_signal
                )
            else:
                clean_attacked, _, _, _ = self.valid_attacks_pipeline.process(
                    original_signal,
                    original_signal
                )
            clean_detector_input = clean_attacked

            attack_info = {
                'applied': True,
                'stats': attack_stats,
                'watermark_coverage': ground_truth_mask[:, 0, :].mean().item() * 100
            }
        else:
            # No attacks: watermark is present everywhere
            detector_input = watermarked_signal
            clean_detector_input = original_signal
            ground_truth_mask = torch.ones(watermarked_signal.size(0), 1, watermarked_signal.size(2),
                                          device=watermarked_signal.device)
            attack_info = {'applied': False}

        # Pipeline: Generator -> Attack -> Locator & Detector (independent)

        locator_output = self.locator(detector_input)
        clean_locator_output = self.locator(clean_detector_input)

        locator_presence_prob = torch.sigmoid(locator_output)  # [B, 1, T]

        watermarked_detector_output = self.detector(detector_input)
        clean_detector_output = self.detector(clean_detector_input)


        output: Dict[str, Any] = {
            'reconstructed_signal': reconstructed_signal,
            'watermarked_signal': watermarked_signal,
            'watermarked_detector_output': watermarked_detector_output,
            'clean_detector_output': clean_detector_output,
            'watermarked_locator_output': locator_output,
            'clean_locator_output': clean_locator_output,
            'ground_truth_mask': ground_truth_mask,
            'attack_info': attack_info,
            'locator_presence_prob': locator_presence_prob
        }

        if phase == self.valid_phase or not self.training:
            effect_key = specific_effect if specific_effect else 'identity'
            results_dict = {
                effect_key: {
                    'watermarked_detector_output': watermarked_detector_output,
                    'clean_detector_output': clean_detector_output,
                    'locator_output': locator_output,
                    'locator_presence_prob': locator_presence_prob,
                }
            }

            if attack_info['applied']:
                results_dict['attacks'] = attack_info

            output['results_dict'] = results_dict
            logger.debug(f"Validation phase: added results_dict to output")

        return output
