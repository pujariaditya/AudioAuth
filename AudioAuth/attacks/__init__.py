"""
Audio watermark attacks module.

This module provides functionality for attacking watermarked audio samples
and generating corresponding ground truth labels for watermark presence detection.
"""

from .localization_attacks import LocalizationAttacks
from .sequence_attacks import SequenceAttacks
from .effect_attacks import EffectAttacks
from .main import AttackPipeline

__all__ = ['LocalizationAttacks', 'SequenceAttacks', 'EffectAttacks', 'AttackPipeline']