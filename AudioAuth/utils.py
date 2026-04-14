"""Audio utilities for the AudioAuth watermarking system.

Provides audio resampling and SNR scaling helpers, model loading/saving
with registry support, data-loading utilities, and convenience functions
for building AudioWatermarking pipelines from configuration.
"""

import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import resampy
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, DistributedSampler

from .config import Config, DatasetConfig, RunConfig
from .dist_utils import get_rank, get_world_size, is_dist_avail_and_initialized
from .models import Generator, Detector, AudioWatermarking
from .models.locator import Locator
from .storage_utils import GSPath, is_gcs_path

logger = logging.getLogger(__name__)


# Constants
TARGET_SAMPLE_RATE = 16_000


# Model registry for different versions and tags
MODEL_REGISTRY = {
    "16khz": {
        "url": None,  # URL for downloading pretrained model
        "description": "16kHz audio tokenizer (AudioAuth configuration)",
        "config": {
            "sample_rate": 16000,
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 5, 8],
            "decoder_dim": 1536,
            "decoder_rates": [8, 5, 4, 2],
            "latent_dim": 256
        }
    },
    "16khz_alt": {
        "url": None,
        "description": "16kHz audio tokenizer (alternative config)",
        "config": {
            "sample_rate": 16000,
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 8, 8],
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 4, 2],
            "latent_dim": 256
        }
    },
    "44khz": {
        "url": None,
        "description": "44.1kHz audio tokenizer",
        "config": {
            "sample_rate": 44100,
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 4, 8],
            "decoder_dim": 1536,
            "decoder_rates": [8, 4, 4, 2],
            "latent_dim": 256
        }
    }
}


def snr_scale(clean: torch.Tensor, noise: torch.Tensor, snr: float) -> torch.Tensor:
    """Scale noise to achieve desired signal-to-noise ratio.
    
    Args:
        clean: Clean signal tensor
        noise: Noise signal tensor (same shape as clean)
        snr: Desired SNR in dB
        
    Returns:
        Scaled noise tensor that achieves the target SNR when added to clean
        
    Raises:
        AssertionError: If clean and noise have different shapes
    """
    assert clean.shape == noise.shape, "Clean and noise must have the same shape."

    power_signal = torch.mean(clean**2)
    power_noise = torch.mean(noise**2)

    epsilon = 1e-10
    power_noise = torch.clamp(power_noise, min=epsilon)

    desired_noise_power = power_signal / (10 ** (snr / 10))

    scale = torch.sqrt(desired_noise_power / power_noise)
    scaled_noise = scale * noise

    return scaled_noise


def time_scale(signal: torch.Tensor, scale: float = 2.0, 
               rngnp: Optional[np.random.Generator] = None, seed: int = 42) -> torch.Tensor:
    """Apply random time scaling to audio signal.
    
    Stretches or compresses the signal in time by a random factor,
    then resamples to original length.
    
    Args:
        signal: Input audio tensor
        scale: Maximum scaling factor (actual scale is random in [1/scale, scale])
        rngnp: NumPy random generator for reproducibility
        seed: Random seed if rngnp not provided
        
    Returns:
        Time-scaled signal with same shape as input
    """
    if rngnp is None:
        rngnp = np.random.default_rng(seed=seed)
    scaling = np.power(scale, rngnp.uniform(-1, 1))
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    # Trim or zero-pad to the original size
    if scaled_signal.shape[-1] > signal.shape[-1]:
        nframes_offset = (scaled_signal.shape[-1] - signal.shape[-1]) // 2
        scaled_signal = scaled_signal[..., nframes_offset : nframes_offset + signal.shape[-1]]
    else:
        nframes_diff = signal.shape[-1] - scaled_signal.shape[-1]
        pad_left = int(np.random.uniform() * nframes_diff)
        pad_right = nframes_diff - pad_left
        scaled_signal = F.pad(input=scaled_signal, pad=(pad_left, pad_right), mode="constant", value=0)
    return scaled_signal


def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Generate mel-scale frequencies.
    
    Args:
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        
    Returns:
        Array of mel frequencies in Hz
    """
    def _hz_to_mel(f: float) -> float:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + f / 700)

    def _mel_to_hz(m: float) -> float:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10 ** (m / 2595) - 1)

    low = _hz_to_mel(fmin)
    high = _hz_to_mel(fmax)

    mels = np.linspace(low, high, n_mels)

    return _mel_to_hz(mels)


def now_as_str() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_device(sample: Any, device: Union[str, torch.device]) -> Any:
    """Move a sample to the specified device.
    
    Args:
        sample: Sample to move (can be nested structure)
        device: Target device ('cuda', 'cpu', or torch.device)
        
    Returns:
        Sample with all tensors moved to device
    """
    def _move_to_device(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_device, sample)


def prepare_sample(samples: Any, cuda_enabled: bool = True) -> Any:
    """Prepare samples for model input.
    
    Args:
        samples: Input samples
        cuda_enabled: Whether to move samples to CUDA
        
    Returns:
        Prepared samples on appropriate device
    """
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_sample_dist(samples: Any, device: Union[str, torch.device]) -> Any:
    """Prepare samples for distributed training.

    Args:
        samples: Input samples
        device: Target device

    Returns:
        Prepared samples on specified device
    """
    samples = move_to_device(samples, device)

    return samples


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def prepare_one_sample(wav_path: str, wav_processor: Optional[Any] = None, 
                      cuda_enabled: bool = True) -> dict:
    """Prepare a single audio sample for inference.

    Args:
        wav_path: Path to the audio file
        wav_processor: Optional function to process the audio
        cuda_enabled: Whether to move the sample to GPU
        
    Returns:
        Dictionary containing processed audio sample
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 10]  # truncate audio to at most 10s

    # spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
    audio_t = torch.tensor(audio).unsqueeze(0)
    audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SAMPLE_RATE)

    samples = {
        "raw_wav": audio_t,
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        "audio_chunk_sizes": [1],
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_one_sample_waveform(audio, cuda_enabled=True, sr=16000):
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 10]  # truncate audio to at most 30s

    samples = {
        "raw_wav": torch.tensor(audio).unsqueeze(0).type(torch.DoubleTensor),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_sample_waveforms(audio_paths, cuda_enabled=True, sr=TARGET_SAMPLE_RATE, max_length_seconds=10):
    batch_len = sr  # minimum length of audio
    audios = []
    for audio_path in audio_paths:
        audio, loaded_sr = sf.read(audio_path)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        audio = audio[: loaded_sr * 10]
        audio = resampy.resample(audio, loaded_sr, sr)
        audio = torch.from_numpy(audio)

        if len(audio) < sr * max_length_seconds:
            pad_size = sr * max_length_seconds - len(audio)
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        audio = torch.clamp(audio, -1.0, 1.0)
        if len(audio) > batch_len:
            batch_len = len(audio)
        audios.append(audio)
    padding_mask = torch.zeros((len(audios), batch_len), dtype=torch.bool)
    for i in range(len(audios)):
        if len(audios[i]) < batch_len:
            pad_len = batch_len - len(audios[i])
            sil = torch.zeros(pad_len, dtype=torch.float32)
            audios[i] = torch.cat((audios[i], sil), dim=0)
            padding_mask[i, len(audios[i]) :] = True
    audios = torch.stack(audios, dim=0)

    samples = {
        "raw_wav": audios,
        "padding_mask": padding_mask,
        "audio_chunk_sizes": [len(audio_paths)],
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def generate_sample_batches(
    audio_path,
    cuda_enabled: bool = True,
    sr: int = TARGET_SAMPLE_RATE,
    chunk_len: int = 10,
    hop_len: int = 5,
    batch_size: int = 4,
):
    audio, loaded_sr = sf.read(audio_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    audio = torchaudio.functional.resample(torch.from_numpy(audio), loaded_sr, sr)
    hop_len = hop_len * sr
    chunk_len = max(len(audio), chunk_len * sr)
    chunks = []

    for i in range(0, len(audio), hop_len):
        chunk = audio[i : i + chunk_len]
        if len(chunk) < chunk_len:
            break
        chunks.append(chunk)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        padding_mask = torch.zeros((len(batch), sr * chunk_len), dtype=torch.bool)
        batch = torch.stack(batch, dim=0)
        samples = {
            "raw_wav": batch,
            "padding_mask": padding_mask,
            "audio_chunk_sizes": [1 for _ in range(len(batch))],
        }
        if cuda_enabled:
            samples = move_to_device(samples, "cuda")
        yield samples


def prepare_samples_for_detection(samples, prompt, label):
    prompts = [prompt for i in range(len(samples["raw_wav"]))]
    labels = [label for i in range(len(samples["raw_wav"]))]
    task = ["detection" for i in range(len(samples["raw_wav"]))]
    samples["prompt"] = prompts
    samples["text"] = labels
    samples["task"] = task
    return samples


def universal_torch_load(
    f: str | os.PathLike,
    *,
    cache_mode: Literal["none", "use", "force"] = "none",
    **kwargs,
) -> Any:
    """
    Wrapper function for torch.load that can handle GCS paths.

    This function provides a convenient way to load PyTorch objects from both local and
    Google Cloud Storage (GCS) paths. For GCS paths, it can optionally caches the
    downloaded files locally to avoid repeated downloads.

    The cache location is determined by:
    1. The ESP_CACHE_HOME environment variable if set
    2. Otherwise defaults to ~/.cache/esp/

    Args:
        f: File-like object, string or PathLike object.
           Can be a local path or a GCS path (starting with 'gs://').
        cache_mode (str, optional): Cache mode for GCS files. Options are:
            "none": No caching (use bucket directly)
            "use": Use cache if available, download if not
            "force": Force redownload even if cache exists
            Defaults to "none".
        **kwargs: Additional keyword arguments passed to torch.load().

    Returns:
        The object loaded from the file using torch.load.

    Raises:
        IsADirectoryError: If the GCS path points to a directory instead of a file.
        FileNotFoundError: If the local file does not exist.
    """

    if is_gcs_path(f):
        gs_path = GSPath(str(f))
        if gs_path.is_dir():
            raise IsADirectoryError(f"Cannot load a directory: {f}")

        if cache_mode in ["use", "force"]:
            if "ESP_CACHE_HOME" in os.environ:
                cache_path = Path(os.environ["ESP_CACHE_HOME"]) / gs_path.name
            else:
                cache_path = Path.home() / ".cache" / "esp" / gs_path.name

            if not cache_path.exists() or cache_mode == "force":
                logger.info(
                    f"{'Force downloading' if cache_mode == 'force' else 'Cache file does not exist, downloading'} to {cache_path}..."
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                gs_path.download_to(cache_path)
            else:
                logger.debug(f"Found {cache_path}, using local cache.")
            f = cache_path
        else:
            f = gs_path
    else:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"File does not exist: {f}")

    with open(f, "rb") as opened_file:
        return torch.load(opened_file, **kwargs)


def download_model(model_tag: str, model_dir: Optional[Path] = None) -> Path:
    """Download a pretrained model if not already cached
    
    Args:
        model_tag: Tag of the model to download
        model_dir: Directory to save the model (default: ~/.cache/audioauth)
        
    Returns:
        Path to the downloaded model file
    """
    if model_tag not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model tag: {model_tag}. Available: {list(MODEL_REGISTRY.keys())}")
    
    if model_dir is None:
        model_dir = Path.home() / ".cache" / "audioauth"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{model_tag}.pt"
    
    if model_path.exists():
        return model_path
    
    url = MODEL_REGISTRY[model_tag]["url"]
    if url is None:
        raise ValueError(f"No pretrained model available for {model_tag}. "
                        "Please train a model or provide a checkpoint path.")
    
    raise NotImplementedError("Model downloading not yet implemented")


def load_model(
    model_path_or_tag: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    load_state_dict: bool = True,
    tag: Optional[str] = None
) -> Generator:
    """Load a tokenizer model from a checkpoint or tag
    
    Args:
        model_path_or_tag: Path to checkpoint file or model tag
        device: Device to load the model on
        load_state_dict: Whether to load the state dict
        tag: Override tag for configuration (if using custom checkpoint)
        
    Returns:
        Loaded Generator model
    """
    model_path_or_tag = str(model_path_or_tag)
    
    if model_path_or_tag in MODEL_REGISTRY:
        tag = model_path_or_tag
        if load_state_dict:
            model_path = download_model(tag)
        else:
            model_path = None
    else:
        model_path = Path(model_path_or_tag)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if tag is None:
            for possible_tag in MODEL_REGISTRY:
                if possible_tag in model_path.stem:
                    tag = possible_tag
                    break
    
    if tag is not None and tag in MODEL_REGISTRY:
        config = MODEL_REGISTRY[tag]["config"]
    else:
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location="cpu")
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                warnings.warn("No config found in checkpoint, using default 16kHz config")
                config = MODEL_REGISTRY["16khz"]["config"]
        else:
            config = MODEL_REGISTRY["16khz"]["config"]
    
    model = Generator(config)

    if load_state_dict and model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


def save_model(
    model: Generator,
    save_path: Union[str, Path],
    config: Optional[Union[Dict, Any]] = None,
    metadata: Optional[Dict] = None
):
    """Save a tokenizer model to a checkpoint file
    
    Args:
        model: Generator model to save
        save_path: Path to save the checkpoint
        config: Configuration to save (uses model.config if None)
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config or model.config,
    }
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, save_path)


def list_models() -> Dict[str, str]:
    """List available model tags and their descriptions
    
    Returns:
        Dictionary mapping tags to descriptions
    """
    return {tag: info["description"] for tag, info in MODEL_REGISTRY.items()}


def get_model_info(tag: str) -> Dict:
    """Get information about a specific model tag
    
    Args:
        tag: Model tag
        
    Returns:
        Dictionary with model information
    """
    if tag not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model tag: {tag}")
    
    info = MODEL_REGISTRY[tag].copy()
    config = info["config"]
    
    info["sample_rate"] = config.sample_rate
    info["downsample_rate"] = config.downsample_rate
    info["frame_rate"] = config.frame_rate
    info["latent_dim"] = config.latent_dim
    
    return info


def load_16khz_model(device: Union[str, torch.device] = "cuda") -> Generator:
    """Load the default 16kHz model
    
    Args:
        device: Device to load the model on
        
    Returns:
        Loaded 16kHz Generator model
    """
    return load_model("16khz", device=device, load_state_dict=False)

def load_discriminator(
    discriminator_type: str = "mpd_msd_mrd",
    period_scales: Optional[List[int]] = None,
    scale_downsample_rates: Optional[List[int]] = None,
    fft_sizes: Optional[List[int]] = None,
    bands: Optional[List[List[float]]] = None,
    sample_rate: int = 16000,
    device: Union[str, torch.device] = "cuda"
) -> nn.Module:
    """Load discriminator model for GAN training.
    
    Args:
        discriminator_type: Type of discriminator ("mpd_msd_mrd", "mpd_msd", "mpd")
        period_scales: Period scales for MPD (default: [2, 3, 5, 7, 11])
        scale_downsample_rates: Downsample rates for MSD (default: [4, 4, 4, 4])
        fft_sizes: FFT sizes for MRD (default: [2048, 1024, 512])
        bands: Frequency bands for MRD
        sample_rate: Sample rate in Hz (default: 16000)
        device: Device to load the model on
        
    Returns:
        Discriminator model
    """
    try:
        from .models.discriminator import Discriminator
    except ImportError:
        raise ImportError(
            "Failed to import discriminator models. "
            "Please ensure the discriminator module is available."
        )
    
    if period_scales is None:
        period_scales = [2, 3, 5, 7, 11]
    if scale_downsample_rates is None:
        scale_downsample_rates = [4, 4, 4, 4]
    if fft_sizes is None:
        fft_sizes = [2048, 1024, 512]
    if bands is None:
        bands = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    
    if discriminator_type == "mpd_msd_mrd":
        discriminator = Discriminator(
            periods=period_scales,
            rates=scale_downsample_rates,
            fft_sizes=fft_sizes,
            sample_rate=sample_rate,
            bands=bands
        )
    elif discriminator_type == "mpd_msd":
        discriminator = Discriminator(
            periods=period_scales,
            rates=scale_downsample_rates,
            fft_sizes=[],  # No MRD
            sample_rate=sample_rate,
            bands=bands
        )
    elif discriminator_type == "mpd":
        discriminator = Discriminator(
            periods=period_scales,
            rates=[],  # No MSD
            fft_sizes=[],  # No MRD
            sample_rate=sample_rate,
            bands=bands
        )
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")
    
    return discriminator.to(device)



def create_model_from_config(
    config: Config,
    device: Union[str, torch.device] = "cuda"
) -> AudioWatermarking:
    """Create a model from a Pydantic configuration.
    
    Args:
        config: Full Config object
        device: Device to create the model on
        
    Returns:
        AudioWatermarking model
        
    Raises:
        ValueError: If config or any required sub-config is None
        TypeError: If config types are incorrect
        RuntimeError: If model creation fails
    """
    if config is None:
        raise ValueError("Config is required - cannot create models without configuration")
    if not hasattr(config, 'watermarking'):
        raise ValueError("Config must have 'watermarking' attribute")
    
    watermark_config = config.watermarking
    if watermark_config is None:
        raise ValueError("config.watermarking is None - cannot create models")
    
    if watermark_config.generator is None:
        raise ValueError("config.watermarking.generator is None - GeneratorConfig required")
    if watermark_config.detector is None:
        raise ValueError("config.watermarking.detector is None - DetectorConfig required")
    if watermark_config.locator is None:
        raise ValueError("config.watermarking.locator is None - LocatorConfig required")
    
    try:
        generator = Generator(watermark_config.generator)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to create Generator: {e}")
    
    try:
        detector = Detector(watermark_config.detector)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to create Detector: {e}")
    
    try:
        locator = Locator(watermark_config.locator)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to create Locator: {e}")
    
    try:
        train_attacks_config = watermark_config.train_attacks.model_dump()
        valid_attacks_config = watermark_config.valid_attacks.model_dump()
        
        model = AudioWatermarking(
            generator=generator,
            detector=detector,
            locator=locator,
            sample_rate=watermark_config.sample_rate,
            train_phase=watermark_config.train_phase,
            valid_phase=watermark_config.valid_phase,
            audio_sample_phase=watermark_config.audio_sample_phase,
            train_attacks_config=train_attacks_config,
            valid_attacks_config=valid_attacks_config
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create AudioWatermarking: {e}")
    
    return model.to(device)


def load_models(
    config: Config,
    checkpoint_path: Optional[Path] = None,
    device: Union[str, torch.device] = "cuda"
) -> Tuple[AudioWatermarking, Optional[nn.Module]]:
    """Load both generator and discriminator models.
    
    Args:
        config: Full configuration object
        checkpoint_path: Optional path to checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (watermarking_system, discriminator) models
    """
    watermarking_system = create_model_from_config(config, device)

    discriminator = None
    if config.discriminator.use_discriminator:
        discriminator = load_discriminator(
            discriminator_type=config.discriminator.discriminator_type,
            period_scales=config.discriminator.period_scales,
            scale_downsample_rates=config.discriminator.scale_downsample_rates,
            fft_sizes=config.discriminator.fft_sizes,
            bands=config.discriminator.bands,
            sample_rate=config.discriminator.sample_rate,
            device=device
        )
    
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "watermarking_system_state_dict" in checkpoint:
            watermarking_system.load_state_dict(checkpoint["watermarking_system_state_dict"])
        else:
            raise KeyError("No watermarking_system_state_dict found in checkpoint")
        
        if discriminator is not None and "discriminator_state_dict" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        if "iteration" in checkpoint:
            print(f"Checkpoint iteration: {checkpoint['iteration']}")
    
    return watermarking_system, discriminator


def load_model_for_inference(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda"
) -> AudioWatermarking:
    """Load model specifically for inference.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        AudioWatermarking model ready for inference
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" not in checkpoint:
        raise KeyError("No config found in checkpoint")

    config_data = checkpoint["config"]
    if not isinstance(config_data, dict):
        raise TypeError(f"Expected config to be a dict, got {type(config_data)}")

    def _convert_paths(obj):
        """Convert PosixPath objects to strings for Pydantic deserialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: _convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_paths(v) for v in obj]
        return obj

    config_data = _convert_paths(config_data)
    config = Config(**config_data)
    model = create_model_from_config(config, device)

    if "watermarking_system_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["watermarking_system_state_dict"])
    else:
        raise KeyError("No watermarking_system_state_dict found in checkpoint")

    model = model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(
    watermarking_system: nn.Module,
    discriminator: Optional[nn.Module] = None
) -> None:
    """Print model summary with detailed component breakdown.
    
    Args:
        watermarking_system: Watermarking system containing generator and detector
        discriminator: Optional discriminator model
    """
    generator = watermarking_system.generator if hasattr(watermarking_system, 'generator') else watermarking_system
    detector = watermarking_system.detector if hasattr(watermarking_system, 'detector') else None
    locator = watermarking_system.locator if hasattr(watermarking_system, 'locator') else None
    
    g_params = count_parameters(generator)
    det_params = count_parameters(detector) if detector is not None else 0
    loc_params = count_parameters(locator) if locator is not None else 0
    
    print(f"\n{'='*50}")
    print(f"Model Parameters Summary:")
    print(f"{'='*50}")
    print(f"Watermarking System:")
    print(f"  Generator Parameters:  {g_params / 1e6:>8.2f}M")
    print(f"  Detector Parameters:   {det_params / 1e6:>8.2f}M")
    print(f"  Locator Parameters:    {loc_params / 1e6:>8.2f}M")
    print(f"  {'─'*35}")
    watermarking_total = g_params + det_params + loc_params
    print(f"  Watermarking Total:    {watermarking_total / 1e6:>8.2f}M")
    
    if discriminator is not None:
        d_params = count_parameters(discriminator)
        print(f"\nDiscriminator:")
        print(f"  Discriminator Parameters: {d_params / 1e6:>5.2f}M")
        print(f"\n{'='*50}")
        print(f"Grand Total Parameters: {(watermarking_total + d_params) / 1e6:>8.2f}M")
    else:
        print(f"\n{'='*50}")
        print(f"Total Parameters:       {watermarking_total / 1e6:>8.2f}M")
    print(f"{'='*50}")
    
    print(f"\nModel Configuration:")
    if hasattr(generator, 'sample_rate'):
        print(f"  Sample Rate: {generator.sample_rate} Hz")
    if hasattr(generator, 'downsample_rate'):
        print(f"  Downsample Rate: {generator.downsample_rate}x")
    elif hasattr(generator, 'encoder_rates'):
        downsample = 1
        for rate in generator.encoder_rates:
            downsample *= rate
        print(f"  Downsample Rate: {downsample}x")
    if hasattr(generator, 'frame_rate'):
        print(f"  Frame Rate: {generator.frame_rate:.1f} Hz")
    if hasattr(generator, 'latent_dim'):
        print(f"  Latent Dimension: {generator.latent_dim}")
    if hasattr(watermarking_system, 'nbits'):
        print(f"  Watermark Bits: {watermarking_system.nbits}")


def ensure_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype.
    
    Args:
        dtype_str: String representation of dtype
        
    Returns:
        torch.dtype object
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    
    return dtype_map[dtype_str]


def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    """Create a DataLoader with optional distributed sampler.

    Args:
        dataset: The dataset to load
        config: Config with batch size and data loading settings
        is_train: Whether this is for training (affects shuffling)
        use_distributed: Whether to use distributed sampler
        
    Returns:
        DataLoader instance (wrapped in IterLoader for training)
    """
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train, num_replicas=get_world_size(), rank=get_rank())
    else:
        sampler = None
    
    collate_fn = getattr(dataset, 'collater', None)
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=False,
        sampler=sampler,
        shuffle=sampler is None and is_train,
        collate_fn=collate_fn,
        drop_last=is_train,
    )
    
    if is_train:
        loader = IterLoader(loader, use_distributed=use_distributed)
    
    return loader