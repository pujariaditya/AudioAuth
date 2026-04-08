"""
Audio Effect Scheduler Module

This module implements an adaptive scheduling system for audio effects that
selects effects and their parameters based on performance metrics (BER and mIoU).
It uses exponential moving averages and success-based weighting to intelligently
adapt effect selection over time.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from AudioAuth.exceptions import (
    EffectSchedulerError,
    InvalidEffectError,
    InvalidMetricError,
    ParameterValidationError
)

logger = logging.getLogger(__name__)


class EffectScheduler:
    """
    Scheduler for audio effects with adaptive selection based on performance metrics.

    This scheduler manages effect probabilities and selects effects based on their
    Bit Error Rate (BER) and mean Intersection over Union (mIoU) performance.
    It uses an exponential moving average to track metrics and adapts selection
    probabilities to favor better-performing effects.

    Attributes:
        effect_params: Dictionary mapping effect names to their parameter configurations.
        beta: Smoothing factor for exponential moving average (0 < beta < 1).
        ber_threshold: Success threshold for BER (lower is better).
        miou_threshold: Success threshold for mIoU (higher is better).
        temperature_start: Initial softmax temperature for probability annealing.
        temperature_end: Final softmax temperature after annealing.
        current_step: Current training step (updated externally via set_training_progress).
        total_steps: Total training steps (updated externally via set_training_progress).
        effect_probabilities: Dict mapping effect names to current selection probabilities.
        parameter_success_rates: Nested dict tracking per-parameter success/failure history,
            keyed by effect name then (param_name, param_value) tuples.
    """

    def __init__(
        self,
        effect_params: Dict[str, Dict[str, Any]],
        beta: float = 0.9,
        ber_threshold: float = 0.001,
        miou_threshold: float = 0.95,
        temperature_start: float = 1.0,
        temperature_end: float = 0.7
    ) -> None:
        """
        Initialize the EffectScheduler with effect parameters and thresholds.

        Args:
            effect_params: Dictionary mapping effect names to their parameter configurations.
                          Each effect can have parameters with 'choices' for random selection.
            beta: Smoothing factor for exponential moving average (0 < beta < 1).
                  Higher values give more weight to historical data.
            ber_threshold: Threshold for considering BER as successful (0 <= threshold <= 1).
            miou_threshold: Threshold for considering mIoU as successful (0 <= threshold <= 1).
            temperature_start: Initial softmax temperature for linear annealing
                (T(t) = T_start - (T_start - T_end) * progress).
            temperature_end: Final softmax temperature after annealing.

        Raises:
            ValueError: If beta is not in range (0, 1) or thresholds are invalid.
            ParameterValidationError: If effect parameters fail validation.
        """
        if not 0 < beta < 1:
            raise ValueError(f"Beta must be in range (0, 1), got {beta}")
        if not 0 <= ber_threshold <= 1:
            raise ValueError(f"BER threshold must be in range [0, 1], got {ber_threshold}")
        if not 0 <= miou_threshold <= 1:
            raise ValueError(f"mIoU threshold must be in range [0, 1], got {miou_threshold}")

        try:
            self._validate_effect_params(effect_params)
        except Exception as e:
            logger.error(f"Effect parameter validation failed: {str(e)}", exc_info=True)
            raise ParameterValidationError(f"Invalid effect parameters: {str(e)}")

        self.effect_params = effect_params
        self.beta = beta
        self.ber_threshold = ber_threshold
        self.miou_threshold = miou_threshold

        num_effects = len(effect_params)
        self.effect_probabilities: Dict[str, float] = {
            effect_name: 1.0 / num_effects for effect_name in effect_params.keys()
        }

        self.effect_usage_stats: Dict[str, int] = {
            name: 0 for name in effect_params.keys()
        }
        self.total_effects: int = 0

        self.effect_metrics_history: Dict[str, Dict[str, Optional[float]]] = {
            effect_name: {'ber': None, 'miou': None}
            for effect_name in effect_params.keys()
        }

        self.current_effect_name: Optional[str] = None

        # Linear temperature annealing: T(t) = T_start - (T_start - T_end) * progress
        self.temperature_start: float = temperature_start
        self.temperature_end: float = temperature_end
        self.current_step: int = 0
        self.total_steps: int = 1  # Will be set by training loop

        self.parameter_success_rates: Dict[str, Dict[Tuple[str, Any], List[bool]]] = {}

        self.effect_list: List[str] = list(effect_params.keys())
        self.effect_ptr: int = 0  # Pointer for effect assignments

        self.parameter_metrics_history: Dict[str, Dict[Any, Dict[str, Any]]] = {
            effect_name: {} for effect_name in effect_params.keys()
        }

        self.metric_history: Dict[str, Dict[str, Any]] = {
            effect_name: {
                'overall': {'ber': [], 'miou': []},
                'params': {}  # Parameter-specific histories
            } for effect_name in effect_params.keys()
        }

    # --- Effect Selection ---

    def select_all_effects(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Return all available effects with selected parameters.

        This method is used when applying all effects to every batch, useful for
        comprehensive testing or when effect diversity is prioritized over
        performance-based selection.

        Returns:
            List of tuples containing (effect_name, selected_parameters) for all
            available effects. Parameters with 'choices' will be randomly selected.

        Raises:
            EffectSchedulerError: If parameter selection fails for any effect.
        """
        effects: List[Tuple[str, Dict[str, Any]]] = []

        try:
            for effect_name in self.effect_params.keys():
                raw_params = self.effect_params.get(effect_name, {})
                self.current_effect_name = effect_name

                effect_params = self._select_effect_params(raw_params)
                effects.append((effect_name, effect_params))

                self.effect_usage_stats[effect_name] += 1
                self.total_effects += 1

            logger.debug(f"Selected all {len(effects)} effects for application")
            return effects

        except Exception as e:
            logger.error(f"Failed to select all effects: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Effect selection failed: {str(e)}")

    def select_effects(self, num_effects: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select effects based on their performance-weighted probabilities.

        This method uses the current effect probabilities (which are adapted based
        on performance) to randomly select a specified number of effects. The same
        effect can be selected multiple times if it has high probability.

        Args:
            num_effects: Number of effects to select. Will be capped at the total
                        number of available effects if larger.

        Returns:
            List of tuples containing (effect_name, selected_parameters) for the
            selected effects. Parameters are selected based on success rates.

        Raises:
            ValueError: If num_effects is not positive.
            EffectSchedulerError: If effect selection or parameter selection fails.
        """
        if num_effects <= 0:
            raise ValueError(f"Number of effects must be positive, got {num_effects}")

        effects: List[Tuple[str, Dict[str, Any]]] = []

        try:
            effect_names = list(self.effect_probabilities.keys())
            probabilities = [self.effect_probabilities[name] for name in effect_names]

            # Normalize probabilities to ensure they sum to 1
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p / prob_sum for p in probabilities]
            else:
                logger.warning("All effect probabilities are 0, using uniform distribution")
                probabilities = [1.0 / len(effect_names) for _ in effect_names]

            selected_names = np.random.choice(
                effect_names,
                size=num_effects,
                replace=True,
                p=probabilities
            )

            for effect_name in selected_names:
                raw_params = self.effect_params.get(effect_name, {})
                self.current_effect_name = effect_name

                effect_params = self._select_effect_params(raw_params)
                effects.append((effect_name, effect_params))

                self.effect_usage_stats[effect_name] += 1
                self.total_effects += 1

            logger.debug(f"Selected {len(effects)} effects based on probabilities")
            return effects

        except Exception as e:
            logger.error(f"Failed to select effects: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Effect selection failed: {str(e)}")

    # --- Metrics and Statistics ---

    def get_effect_probabilities(self) -> Dict[str, float]:
        """
        Get current effect selection probabilities.

        Returns:
            Dictionary mapping effect names to their current selection probabilities.
            Probabilities sum to 1.0 and reflect the adaptive weighting based on
            performance metrics.
        """
        return self.effect_probabilities.copy()

    def get_effect_statistics(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Get comprehensive statistics for all effects.

        Returns:
            Dictionary mapping effect names to their statistics including:
            - usage_percentage: Percentage of total selections
            - ema_ber: Exponential moving average of BER (if available)
            - ema_miou: Exponential moving average of mIoU (if available)
            - avg_ber: Simple average of all BER measurements (if available)
            - avg_miou: Simple average of all mIoU measurements (if available)
            - selection_count: Total number of times selected
        """
        stats: Dict[str, Dict[str, Optional[float]]] = {}

        try:
            for effect_name in self.effect_params.keys():
                metrics = self.effect_metrics_history[effect_name]

                usage_pct = (
                    (self.effect_usage_stats[effect_name] / self.total_effects * 100)
                    if self.total_effects > 0 else 0.0
                )

                history = self.metric_history[effect_name]['overall']
                avg_ber = np.mean(history['ber']) if history['ber'] else None
                avg_miou = np.mean(history['miou']) if history['miou'] else None

                stats[effect_name] = {
                    'usage_percentage': usage_pct,
                    'ema_ber': metrics['ber'],
                    'ema_miou': metrics['miou'],
                    'avg_ber': avg_ber,
                    'avg_miou': avg_miou,
                    'selection_count': self.effect_usage_stats[effect_name]
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get effect statistics: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Failed to get effect statistics: {str(e)}")

    def update_effect_metrics(
        self,
        effect_name: str,
        effect_params: Dict[str, Any],
        localized_ber: float,
        miou: float
    ) -> None:
        """
        Update performance metrics for a specific effect and parameter combination.

        This method updates both the overall effect metrics and parameter-specific
        metrics using exponential moving averages. It also tracks success rates
        for individual parameter values to enable adaptive parameter selection.

        Args:
            effect_name: Name of the effect to update metrics for.
            effect_params: Dictionary of parameter values used for this measurement.
            localized_ber: Localized Bit Error Rate measurement (computed only in watermarked regions, 0 <= BER <= 1, lower is better).
            miou: Mean Intersection over Union measurement (0 <= mIoU <= 1, higher is better).

        Raises:
            InvalidEffectError: If effect_name is not recognized.
            InvalidMetricError: If metric values are outside valid range [0, 1].
        """
        if effect_name not in self.effect_params:
            raise InvalidEffectError(f"Unknown effect: '{effect_name}'")

        if not 0 <= localized_ber <= 1:
            raise InvalidMetricError(
                f"BER must be in range [0, 1], got {localized_ber}"
            )

        if not 0 <= miou <= 1:
            raise InvalidMetricError(
                f"mIoU must be in range [0, 1], got {miou}"
            )

        try:
            beta = self.beta

            metrics = self.effect_metrics_history.setdefault(
                effect_name, {'ber': None, 'miou': None}
            )

            if metrics['ber'] is None:
                self.effect_metrics_history[effect_name]['ber'] = localized_ber
            else:
                self.effect_metrics_history[effect_name]['ber'] = (
                    beta * metrics['ber'] + (1 - beta) * localized_ber
                )

            if metrics['miou'] is None:
                self.effect_metrics_history[effect_name]['miou'] = miou
            else:
                self.effect_metrics_history[effect_name]['miou'] = (
                    beta * metrics['miou'] + (1 - beta) * miou
                )

            effect_history = self.metric_history[effect_name]
            effect_history['overall']['ber'].append(localized_ber)
            effect_history['overall']['miou'].append(miou)

            param_key = self.make_hashable(effect_params)
            if param_key not in effect_history['params']:
                effect_history['params'][param_key] = {'ber': [], 'miou': []}

            effect_history['params'][param_key]['ber'].append(localized_ber)
            effect_history['params'][param_key]['miou'].append(miou)

            is_success = (
                localized_ber <= self.ber_threshold and
                miou >= self.miou_threshold
            )

            # Special handling for bandpass frequency pairs
            param_tuple = None
            if effect_name == 'bandpass' and 'frequency_pairs' in effect_params:
                freq_pairs_config = effect_params['frequency_pairs']
                if 'choices' in freq_pairs_config and freq_pairs_config['choices']:
                    pair = freq_pairs_config['choices'][0]
                    pair_hashable = self.make_hashable(pair)
                    param_tuple = ('frequency_pairs', pair_hashable)
            elif effect_name == 'bandpass' and '_frequency_pair' in effect_params:
                pair = effect_params['_frequency_pair']
                pair_hashable = self.make_hashable(pair)
                param_tuple = ('frequency_pairs', pair_hashable)

            if param_tuple is not None:
                if effect_name not in self.parameter_success_rates:
                    self.parameter_success_rates[effect_name] = {}
                if param_tuple not in self.parameter_success_rates[effect_name]:
                    self.parameter_success_rates[effect_name][param_tuple] = []

                self.parameter_success_rates[effect_name][param_tuple].append(is_success)
            else:
                if effect_params:
                    for param_name, param_value in effect_params.items():
                        if param_name.startswith('_'):
                            continue

                        param_value_hashable = self.make_hashable(param_value)
                        param_tuple = (param_name, param_value_hashable)

                        if effect_name not in self.parameter_success_rates:
                            self.parameter_success_rates[effect_name] = {}
                        if param_tuple not in self.parameter_success_rates[effect_name]:
                            self.parameter_success_rates[effect_name][param_tuple] = []

                        self.parameter_success_rates[effect_name][param_tuple].append(is_success)
                else:
                    param_tuple = ('no_params', None)

                    if effect_name not in self.parameter_success_rates:
                        self.parameter_success_rates[effect_name] = {}
                    if param_tuple not in self.parameter_success_rates[effect_name]:
                        self.parameter_success_rates[effect_name][param_tuple] = []

                    self.parameter_success_rates[effect_name][param_tuple].append(is_success)

            param_metrics = self.parameter_metrics_history[effect_name].setdefault(
                param_key, {'ber': None, 'miou': None, 'count': 0}
            )

            if param_metrics['ber'] is None:
                param_metrics['ber'] = localized_ber
                param_metrics['miou'] = miou
            else:
                param_metrics['ber'] = beta * param_metrics['ber'] + (1 - beta) * localized_ber
                param_metrics['miou'] = beta * param_metrics['miou'] + (1 - beta) * miou

            param_metrics['count'] += 1

            logger.debug(
                f"Updated metrics for {effect_name}: BER={localized_ber:.4f}, "
                f"mIoU={miou:.4f}, success={is_success}"
            )

        except Exception as e:
            logger.error(f"Failed to update effect metrics: {str(e)}", exc_info=True)
            raise InvalidMetricError(f"Failed to update effect metrics: {str(e)}")

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        """
        Update training progress for temperature annealing.

        Temperature anneals linearly: T(t) = T_start - (T_start - T_end) * progress.

        Args:
            current_step: Current training step.
            total_steps: Total number of training steps.
        """
        self.current_step = current_step
        self.total_steps = max(total_steps, 1)

    def _get_temperature(self) -> float:
        """
        Compute current temperature based on training progress.

        Temperature anneals linearly from temperature_start to temperature_end
        over the course of training: T(t) = T_start - (T_start - T_end) * progress.

        Returns:
            Current temperature value, clamped to [temperature_end, temperature_start].
        """
        progress = min(self.current_step / self.total_steps, 1.0)
        temperature = self.temperature_start - (self.temperature_start - self.temperature_end) * progress
        return max(self.temperature_end, min(self.temperature_start, temperature))

    def adapt_effect_probabilities(self) -> None:
        """
        Adapt effect selection probabilities based on WORST performance metrics.

        This method recalculates effect probabilities using a penalty-based system
        where effects with higher BER and lower mIoU receive higher selection
        probabilities (selecting worst performers). The adaptation uses exponential
        smoothing to prevent sudden changes and maintain stability.

        The penalty calculation gives 80% weight to BER performance (higher BER =
        higher probability) and 20% weight to inverted mIoU performance (lower mIoU =
        higher probability).

        Raises:
            EffectSchedulerError: If probability adaptation fails.
        """
        try:
            effect_scores: Dict[str, float] = {}
            smoothing_factor = 0.8

            for effect_name, param_metrics in self.parameter_metrics_history.items():
                if not param_metrics:
                    effect_scores[effect_name] = 0.0
                    continue

                param_scores: List[float] = []
                for metrics in param_metrics.values():
                    if metrics['ber'] is not None and metrics['miou'] is not None:
                        # Higher BER is worse, lower mIoU is worse; penalize both
                        penalty = 0.8 * metrics['ber'] + 0.2 * (1 - metrics['miou'])
                        param_scores.append(penalty)

                if param_scores:
                    effect_scores[effect_name] = np.mean(param_scores)
                else:
                    effect_scores[effect_name] = 0.0

            effect_names = list(effect_scores.keys())
            scores = np.array([effect_scores[name] for name in effect_names])

            if np.all(scores == 0):
                new_probabilities = np.ones_like(scores) / len(scores)
                logger.debug("All effect scores are zero, maintaining uniform distribution")
            else:
                # Softmax with linearly annealed temperature:
                # T(t) = T_start - (T_start - T_end) * progress
                temperature = self._get_temperature()
                scores_stable = scores - np.max(scores)  # Prevent overflow
                exp_scores = np.exp(scores_stable / temperature)
                new_probabilities = exp_scores / np.sum(exp_scores)

            # Exponential smoothing prevents sudden probability jumps
            for effect_name, new_prob in zip(effect_names, new_probabilities):
                old_prob = self.effect_probabilities[effect_name]
                smoothed_prob = (
                    smoothing_factor * old_prob +
                    (1 - smoothing_factor) * new_prob
                )
                self.effect_probabilities[effect_name] = smoothed_prob

            self._normalize_probabilities()

            logger.debug("Effect probabilities adapted based on performance metrics")

        except Exception as e:
            logger.error(f"Failed to adapt effect probabilities: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Probability adaptation failed: {str(e)}")

    def log_adaptive_behavior(self, logger_func: Optional[Any] = None) -> None:
        """
        Log comprehensive adaptive behavior statistics.

        This method outputs detailed information about the current state of the
        scheduler including effect probabilities, performance metrics, and usage
        statistics. Useful for monitoring and debugging adaptive behavior.

        Args:
            logger_func: Optional logging function to use. If None, uses print().
                        The function should accept a single string argument.
        """
        if logger_func is None:
            logger_func = print

        try:
            logger_func("\n" + "=" * 60)
            logger_func("EFFECT SCHEDULER ADAPTIVE BEHAVIOR")
            logger_func("=" * 60)

            logger_func("\nEffect Selection Probabilities:")
            for effect, prob in sorted(
                self.effect_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger_func(f"  {effect}: {prob:.4f}")

            stats = self.get_effect_statistics()
            logger_func("\nEffect Performance Statistics:")
            for effect, effect_stats in sorted(stats.items()):
                logger_func(f"\n  {effect}:")
                logger_func(f"    Usage: {effect_stats['usage_percentage']:.1f}%")

                if effect_stats['ema_ber'] is not None:
                    logger_func(f"    EMA BER: {effect_stats['ema_ber']:.4f}")
                if effect_stats['ema_miou'] is not None:
                    logger_func(f"    EMA mIoU: {effect_stats['ema_miou']:.4f}")
                if effect_stats['avg_ber'] is not None:
                    logger_func(f"    Avg BER: {effect_stats['avg_ber']:.4f}")
                if effect_stats['avg_miou'] is not None:
                    logger_func(f"    Avg mIoU: {effect_stats['avg_miou']:.4f}")

            logger_func("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Failed to log adaptive behavior: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Failed to log adaptive behavior: {str(e)}")

    # --- Parameter Handling ---

    def _validate_effect_params(self, effect_params: Dict[str, Dict[str, Any]]) -> None:
        """
        Validate effect parameters with strict requirements for bandpass.

        This method ensures that effect parameters are properly structured.
        For bandpass, frequency_pairs is REQUIRED and old format is rejected.

        Args:
            effect_params: Dictionary of effect parameters to validate.

        Raises:
            ParameterValidationError: If validation fails.
        """
        try:
            if 'bandpass' in effect_params:
                bp_params = effect_params['bandpass']

                if 'low_cutoff' in bp_params or 'high_cutoff' in bp_params:
                    raise ParameterValidationError(
                        "Bandpass filter using old format (low_cutoff/high_cutoff). "
                        "Must use frequency_pairs format instead."
                    )

                if 'frequency_pairs' not in bp_params:
                    raise ParameterValidationError(
                        "Bandpass filter missing required 'frequency_pairs' parameter. "
                        "Configuration must include frequency_pairs with choices list."
                    )

                freq_pairs_config = bp_params.get('frequency_pairs', {})
                if not isinstance(freq_pairs_config, dict) or 'choices' not in freq_pairs_config:
                    raise ParameterValidationError(
                        "Invalid frequency_pairs format. Expected dict with 'choices' key."
                    )

                freq_pairs = freq_pairs_config.get('choices', [])
                if not freq_pairs:
                    raise ParameterValidationError(
                        "Bandpass frequency_pairs choices list cannot be empty."
                    )

                for i, pair in enumerate(freq_pairs):
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ParameterValidationError(
                            f"Invalid frequency pair at index {i}: {pair}. "
                            f"Expected [low_freq, high_freq]"
                        )

                    low_freq, high_freq = pair
                    if not isinstance(low_freq, (int, float)) or not isinstance(high_freq, (int, float)):
                        raise ParameterValidationError(
                            f"Frequency values must be numeric. Got pair: {pair}"
                        )

                    if low_freq >= high_freq:
                        raise ParameterValidationError(
                            f"Invalid frequency pair [{low_freq}, {high_freq}]: "
                            f"low_cutoff must be less than high_cutoff"
                        )

                logger.debug(f"Validated {len(freq_pairs)} bandpass frequency pairs")

        except ParameterValidationError:
            raise
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}", exc_info=True)
            raise ParameterValidationError(f"Failed to validate parameters: {str(e)}")

    def _select_effect_params(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select effect parameters from choices based on failure rates (WORST PERFORMING).

        For parameters with 'choices', this method uses historical failure rates
        to weight the selection probability. Parameters that have led to more
        FAILED outcomes (high BER, low mIoU) are more likely to be selected.
        Special handling for paired parameters like bandpass frequency_pairs.

        Args:
            raw_params: Raw parameter configuration which may contain 'choices'.

        Returns:
            Selected parameters with specific values chosen from choices.

        Raises:
            ParameterValidationError: If parameter selection fails.
        """
        selected_params: Dict[str, Any] = {}

        try:
            if self.current_effect_name == 'bandpass':
                if 'frequency_pairs' not in raw_params:
                    raise ParameterValidationError(
                        f"Bandpass effect missing required 'frequency_pairs' parameter"
                    )

                freq_pairs_config = raw_params.get('frequency_pairs', {})
                if not isinstance(freq_pairs_config, dict) or 'choices' not in freq_pairs_config:
                    raise ParameterValidationError(
                        f"Invalid frequency_pairs format for bandpass. Expected dict with 'choices' key."
                    )

                choices_list = freq_pairs_config['choices']
                if not choices_list:
                    raise ParameterValidationError(
                        f"Empty frequency pairs choices list for bandpass"
                    )

                # Weight by historical failure rate so the hardest parameters are trained more
                weights: List[float] = []
                for pair in choices_list:
                    pair_hashable = self.make_hashable(pair)
                    param_tuple = ('frequency_pairs', pair_hashable)

                    success_history = (
                        self.parameter_success_rates
                        .get(self.current_effect_name, {})
                        .get(param_tuple, [])
                    )

                    if success_history:
                        failure_rate = 1 - (sum(success_history) / len(success_history))
                    else:
                        failure_rate = 0.5  # Neutral weight for unexplored pairs

                    # Small constant avoids zero weights
                    weight = failure_rate + 0.1
                    weights.append(weight)

                total_weight = sum(weights)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in weights]
                    choice_idx = np.random.choice(len(choices_list), p=probabilities)
                else:
                    choice_idx = np.random.randint(len(choices_list))

                selected_pair = choices_list[choice_idx]
                selected_params['frequency_pairs'] = {
                    'choices': [selected_pair]
                }

                return selected_params

            for param_key, param_config in raw_params.items():
                if param_key == 'frequency_pairs':
                    continue
                if isinstance(param_config, dict) and 'choices' in param_config:
                    choices_list = param_config['choices']

                    if not choices_list:
                        logger.warning(f"Empty choices list for parameter {param_key}")
                        continue

                    # Weight by historical failure rate so the hardest parameters are trained more
                    weights: List[float] = []
                    for choice in choices_list:
                        choice_hashable = self.make_hashable(choice)
                        param_tuple = (param_key, choice_hashable)

                        success_history = (
                            self.parameter_success_rates
                            .get(self.current_effect_name, {})
                            .get(param_tuple, [])
                        )

                        if success_history:
                            failure_rate = 1 - (sum(success_history) / len(success_history))
                        else:
                            failure_rate = 0.5  # Neutral weight for unexplored values

                        # Small constant avoids zero weights
                        weight = failure_rate + 0.1
                        weights.append(weight)

                    total_weight = sum(weights)
                    if total_weight > 0:
                        probabilities = [w / total_weight for w in weights]
                        choice_idx = np.random.choice(len(choices_list), p=probabilities)
                    else:
                        choice_idx = np.random.randint(len(choices_list))

                    selected_params[param_key] = choices_list[choice_idx]

                else:
                    selected_params[param_key] = param_config

            return selected_params

        except Exception as e:
            logger.error(f"Failed to select parameters: {str(e)}", exc_info=True)
            raise ParameterValidationError(f"Parameter selection failed: {str(e)}")

    # --- Utilities ---

    def _normalize_probabilities(self) -> None:
        """
        Normalize effect probabilities to sum to 1.0 with numerical stability.

        This method handles edge cases like very small probability sums and
        ensures the final probabilities are properly normalized even in the
        presence of numerical errors.

        Raises:
            EffectSchedulerError: If normalization fails completely.
        """
        try:
            total = sum(self.effect_probabilities.values())

            if abs(total - 1.0) > 1e-6:
                total = max(total, 1e-10)

                for key in self.effect_probabilities:
                    self.effect_probabilities[key] /= total

            final_total = sum(self.effect_probabilities.values())
            if abs(final_total - 1.0) > 1e-6:
                logger.warning(
                    f"Probability normalization failed (sum={final_total}), "
                    f"using uniform distribution"
                )
                num_effects = len(self.effect_probabilities)
                for key in self.effect_probabilities:
                    self.effect_probabilities[key] = 1.0 / num_effects

        except Exception as e:
            logger.error(f"Failed to normalize probabilities: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Probability normalization failed: {str(e)}")

    def make_hashable(self, value: Any) -> Any:
        """
        Convert a value to a hashable representation for dictionary keys.

        This method recursively converts lists, tuples, dicts, and numpy arrays
        to hashable tuple representations that can be used as dictionary keys.

        Args:
            value: Any value to convert to hashable form.

        Returns:
            Hashable representation of the input value.
        """
        if isinstance(value, (list, tuple)):
            return tuple(self.make_hashable(v) for v in value)
        elif isinstance(value, dict):
            return tuple(sorted((k, self.make_hashable(v)) for k, v in value.items()))
        elif isinstance(value, np.ndarray):
            return tuple(value.tolist())
        else:
            return value
