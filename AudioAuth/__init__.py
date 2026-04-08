"""AudioAuth: Watermarking Framework with Dual-Purpose Encoding for Localized Robustness and Integrity

A comprehensive audio watermarking framework that combines robustness and localization capabilities.
"""

from .config import Config
from .runner import Runner
from .utils import load_models, print_model_summary

__all__ = [
    # Core classes
    "Config",
    "Runner",
    
    # Model utilities
    "load_models",
    "print_model_summary",
]