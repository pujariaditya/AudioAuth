"""AudioAuth Model Components

This module provides all the core components for the AudioAuth framework:
- Encoder and decoder networks
- Vector quantization modules
- Discriminator networks for adversarial training
- Custom layers and activation functions
"""

from .modules.seanet import SEANetEncoder, SEANetDecoder
from .discriminator import Discriminator, MPD, MSD, MRD
from .layers import WNConv1d, WNConvTranspose1d, Snake1d
from .watermarking import AudioWatermarking
from .generator import Generator
from .detector import Detector
from .locator import Locator

__all__ = [
    "SEANetEncoder",
    "SEANetDecoder", 
    "Discriminator",
    "MPD",
    "MSD",
    "MRD",
    "WNConv1d",
    "WNConvTranspose1d",
    "Snake1d",
    "AudioWatermarking",
    "Generator",
    "Detector",
    "Locator",
]