"""Audio processors for watermarking system.

Provides audio loading utilities for the watermarking-based architecture.
"""

import warnings
from pathlib import Path
from typing import Union
import subprocess
import tempfile

import numpy as np
import resampy
import soundfile as sf
import torch
import torchaudio


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    device: str = "cpu",
    use_ffmpeg: bool = False
) -> torch.Tensor:
    """Load audio file and preprocess.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize to [-1, 1] if True
        device: Device to place tensor on
        use_ffmpeg: Use ffmpeg for loading (for problematic files)
        
    Returns:
        Audio tensor of shape (C, T) where C=1 for mono
    """
    audio_path = Path(audio_path)
    
    if use_ffmpeg:
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            cmd = [
                'ffmpeg', '-i', str(audio_path),
                '-ar', str(sample_rate),
                '-ac', '1' if mono else '2',
                '-f', 'wav',
                tmp_file.name,
                '-y'
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            audio_tensor, sr = torchaudio.load(tmp_file.name)
    else:
        try:
            audio_tensor, sr = torchaudio.load(audio_path)
        except Exception as e:
            try:
                audio_np, sr = sf.read(audio_path, dtype='float32')
                if audio_np.ndim == 2:
                    audio_np = audio_np.T  # (samples, channels) -> (channels, samples)
                else:
                    audio_np = audio_np[np.newaxis, :]
                audio_tensor = torch.from_numpy(audio_np)
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio with torchaudio ({e}) and soundfile ({e2})")
    
    if sr != sample_rate:
        if hasattr(torchaudio.transforms, 'Resample'):
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio_tensor = resampler(audio_tensor)
        else:
            audio_np = audio_tensor.numpy()
            if audio_np.shape[0] == 1:
                audio_np = resampy.resample(audio_np[0], sr, sample_rate)
                audio_np = audio_np[np.newaxis, :]
            else:
                audio_np = np.stack([
                    resampy.resample(audio_np[i], sr, sample_rate)
                    for i in range(audio_np.shape[0])
                ])
            audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
    
    if mono and audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    
    if normalize:
        max_val = audio_tensor.abs().max()
        if max_val > 1.0:
            warnings.warn(f"Audio has values outside [-1, 1] range (max: {max_val:.3f}). Clamping.")
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    
    return audio_tensor.to(device)