"""Inference pipeline for DualMark audio watermarking.

Generates watermarked audio files from a trained checkpoint and computes
quality metrics (SI-SNR) against the originals.
"""

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import torch
import torchaudio

from AudioAuth.config import Config
from AudioAuth.processor import load_audio
from AudioAuth.utils import create_model_from_config
from AudioAuth.models import AudioWatermarking

logger = logging.getLogger(__name__)


def adapt_state_dict(old_sd: Dict[str, torch.Tensor], model_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Adapt a legacy checkpoint state dict to the current model architecture.

    Handles two legacy formats:
      v1 (early): single ``film_layers``, single ``msg_embedding`` (32-bit input),
                  ``gamma_layer``/``beta_layer`` naming.
      v2 (mid):   split ``model_film_layers``/``data_film_layers``,
                  split ``model_embedding``/``data_embedding``,
                  ``gamma_layer``/``beta_layer`` naming, stale ``film_layers`` remnant.

    Current code expects ``delta_gamma_layer``/``delta_beta_layer`` naming,
    ``embedding_norm``/``embedding_gate`` layers (left at default init), and
    ``locator.last_layer`` with output_channels=1 (checkpoint may have 2).
    """
    has_model_film = any("model_film_layers" in k for k in old_sd)
    has_old_film = any(
        "film_layers" in k and "model_film" not in k and "data_film" not in k
        for k in old_sd
    )
    has_msg_embed = any("msg_embedding" in k for k in old_sd)

    new_sd = OrderedDict()

    for key, val in old_sd.items():
        new_key = key

        # --- v1: single msg_embedding (32-bit) → split into model/data ---
        if has_msg_embed and "msg_embedding" in key:
            # First linear layer has input dim 32 → split columns
            for prefix in ("model_embedding", "data_embedding"):
                split_key = key.replace("msg_embedding", prefix)
                split_key = _rename_film(split_key)
                if ".0.weight" in key:
                    half = val.shape[1] // 2
                    if prefix == "model_embedding":
                        new_sd[split_key] = val[:, :half]
                    else:
                        new_sd[split_key] = val[:, half:]
                else:
                    new_sd[split_key] = val.clone()
            continue

        # --- v1: single film_layers → duplicate to model + data film_layers ---
        if has_msg_embed and has_old_film and not has_model_film:
            if "film_layers" in key and "model_film" not in key and "data_film" not in key:
                # Old format: film_layers.{block}.{band}.gamma_layer
                # But v1 old format was: film_layers.{block}.{band}.gamma_layer
                for prefix in ("model_film_layers", "data_film_layers"):
                    dup_key = key.replace("film_layers", prefix)
                    dup_key = _rename_film(dup_key)
                    new_sd[dup_key] = val.clone()
                continue

        # --- v2: drop stale single film_layers remnant ---
        if has_model_film and has_old_film:
            if "film_layers" in key and "model_film" not in key and "data_film" not in key:
                continue  # drop remnant

        # --- Rename gamma_layer/beta_layer → delta_gamma_layer/delta_beta_layer ---
        new_key = _rename_film(new_key)

        # --- Truncate locator.last_layer if output_channels changed (2 → 1) ---
        if new_key in model_sd and val.shape != model_sd[new_key].shape:
            if "last_layer" in new_key:
                target_shape = model_sd[new_key].shape
                if val.dim() >= 1 and val.shape[0] > target_shape[0]:
                    val = val[:target_shape[0]]
                    logger.info(f"Truncated {new_key} from {old_sd[key].shape} to {val.shape}")

        new_sd[new_key] = val

    return new_sd


def _init_gates_passthrough(model: AudioWatermarking) -> None:
    """Set embedding gate outputs to ~1.0 for legacy checkpoints without gates.

    Finds every ``*_embedding_gate`` submodule and sets the last Linear
    layer's bias to +10 (so sigmoid ≈ 1.0) and weight to ~0 (so the
    gate ignores input and always passes through).
    """
    for name, module in model.named_modules():
        if "embedding_gate" in name and isinstance(module, torch.nn.Sequential):
            # Find the last Linear in the gate Sequential
            for child in reversed(list(module.children())):
                if isinstance(child, torch.nn.Linear):
                    torch.nn.init.zeros_(child.weight)
                    torch.nn.init.constant_(child.bias, 10.0)
                    logger.info(f"Initialized {name} last linear bias=10.0 for pass-through")
                    break


def _rename_film(key: str) -> str:
    """Rename gamma_layer → delta_gamma_layer and beta_layer → delta_beta_layer."""
    key = re.sub(r"(?<!delta_)gamma_layer", "delta_gamma_layer", key)
    key = re.sub(r"(?<!delta_)beta_layer", "delta_beta_layer", key)
    return key


def load_watermarking_system(
    config_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
) -> Tuple[AudioWatermarking, Config]:
    """Load a trained watermarking model from config + checkpoint.

    Uses Config.from_yaml to build the model architecture, then loads
    the trained weights from the checkpoint's state_dict. Automatically
    adapts legacy checkpoint formats to the current architecture.

    Args:
        config_path: Path to YAML config used during training.
        checkpoint_path: Path to .pth checkpoint file.
        device: Device to place the model on.

    Returns:
        (model, config) tuple with the model in eval mode.
    """
    config = Config.from_yaml(config_path)
    model = create_model_from_config(config, device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_sd = checkpoint["watermarking_system_state_dict"]
    model_sd = model.state_dict()

    # Try direct load first; fall back to adaptation
    try:
        model.load_state_dict(ckpt_sd)
    except RuntimeError:
        logger.info("Direct state_dict load failed, adapting legacy checkpoint keys...")
        adapted_sd = adapt_state_dict(ckpt_sd, model_sd)
        missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
        if missing:
            logger.info(f"Keys left at default init ({len(missing)}): "
                        f"{[k for k in missing if 'embedding_norm' in k or 'embedding_gate' in k][:6]}...")
        if unexpected:
            logger.warning(f"Unexpected keys dropped ({len(unexpected)}): {unexpected[:5]}...")

        # Initialize embedding gates to pass-through (sigmoid(10) ≈ 1.0)
        # so that legacy checkpoints without gates behave identically.
        _init_gates_passthrough(model)

    model = model.to(device)
    model.eval()

    iteration = checkpoint.get("iteration", "unknown")
    logger.info(f"Loaded checkpoint from iteration {iteration}")
    return model, config


def encode_audio(
    model: AudioWatermarking,
    audio: torch.Tensor,
    msg: torch.Tensor,
    phase: str,
) -> Dict[str, torch.Tensor]:
    """Run a single forward pass in audio_sample phase (no attacks).

    Args:
        model: Trained AudioWatermarking model.
        audio: Audio tensor [B, 1, T].
        msg: Message tensor [B, nbits].
        phase: Phase identifier (should be audio_sample_phase).

    Returns:
        Model output dict containing 'watermarked_signal' etc.
    """
    with torch.no_grad():
        output = model(audio, msg, phase=phase)
    return output


def compute_sisnr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-noise ratio (SI-SNR) in dB.

    Args:
        reference: Clean signal [1, 1, T].
        estimate: Watermarked signal [1, 1, T].
        eps: Small constant for numerical stability.

    Returns:
        SI-SNR value in dB (higher is better).
    """
    ref = reference.squeeze()
    est = estimate.squeeze()
    ref_energy = torch.dot(ref, ref) + eps
    proj = torch.dot(est, ref) / ref_energy * ref
    noise = est - proj
    sisnr = 10 * torch.log10((torch.dot(proj, proj) + eps) / (torch.dot(noise, noise) + eps))
    return sisnr.item()


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Path to YAML config file (e.g. configs/train_stage2.yml)")
@click.option("--checkpoint", "checkpoint_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Path to .pth checkpoint file")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Path to JSONL manifest (id, audio_path, duration)")
@click.option("--output-dir", default="outputs/inference", type=click.Path(path_type=Path),
              help="Directory for output WAV files and results.json")
@click.option("--num-samples", default=10, type=int, help="Number of samples to process")
@click.option("--device", default="cuda", type=str, help="Device (cuda or cpu)")
@click.option("--max-duration", default=10.0, type=float,
              help="Maximum audio duration in seconds (longer clips are truncated)")
@click.option("--seed", default=42, type=int, help="Random seed for reproducible data bits")
def run_inference(
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    num_samples: int,
    device: str,
    max_duration: float,
    seed: int,
) -> None:
    """Generate watermarked audio from a trained DualMark checkpoint.

    For each sample in the manifest, embeds a 32-bit message (16 fixed
    model bits + 16 random data bits), saves original and watermarked
    WAV files, and reports SI-SNR quality metrics.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.manual_seed(seed)

    # Load model
    model, config = load_watermarking_system(config_path, checkpoint_path, device)
    wm_config = config.watermarking
    sample_rate = wm_config.sample_rate
    max_samples = int(max_duration * sample_rate)

    logger.info(f"Model pattern: {wm_config.model_pattern}")
    logger.info(f"Sample rate: {sample_rate}, max duration: {max_duration}s")

    # Prepare output dirs
    orig_dir = output_dir / "original"
    wm_dir = output_dir / "watermarked"
    orig_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    with open(manifest_path) as f:
        entries = [json.loads(line) for line in f]
    entries = entries[:num_samples]
    logger.info(f"Processing {len(entries)} samples from {manifest_path}")

    # Build fixed model bits tensor
    model_bits = torch.tensor(wm_config.model_pattern, dtype=torch.float32)

    results = []
    for entry in entries:
        sample_id = entry["id"]
        audio_path = entry["audio_path"]

        # Load and prepare audio
        audio = load_audio(audio_path, sample_rate=sample_rate, mono=True, device=device)
        if audio.shape[-1] > max_samples:
            audio = audio[:, :max_samples]
        audio = audio.unsqueeze(0)  # [1, 1, T]

        # Construct 32-bit message: [model_bits | random_data_bits]
        data_bits = torch.randint(0, 2, (wm_config.data_bits,), dtype=torch.float32)
        msg = torch.cat([model_bits, data_bits]).unsqueeze(0).to(device)  # [1, 32]

        # Forward pass
        output = encode_audio(model, audio, msg, phase=wm_config.audio_sample_phase)
        watermarked = output["watermarked_signal"]

        # Compute quality
        sisnr = compute_sisnr(audio, watermarked)

        # Save WAV files (move to CPU for torchaudio)
        orig_cpu = audio.squeeze(0).cpu()  # [1, T]
        wm_cpu = watermarked.squeeze(0).cpu()  # [1, T]
        torchaudio.save(str(orig_dir / f"{sample_id}.wav"), orig_cpu, sample_rate)
        torchaudio.save(str(wm_dir / f"{sample_id}.wav"), wm_cpu, sample_rate)

        results.append({"id": sample_id, "sisnr_db": round(sisnr, 2), "duration_s": round(audio.shape[-1] / sample_rate, 2)})
        logger.info(f"  {sample_id}: SI-SNR = {sisnr:.2f} dB, duration = {audio.shape[-1] / sample_rate:.2f}s")

    # Aggregate and save results
    avg_sisnr = sum(r["sisnr_db"] for r in results) / len(results) if results else 0.0
    summary = {"num_samples": len(results), "avg_sisnr_db": round(avg_sisnr, 2), "samples": results}

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Average SI-SNR: {avg_sisnr:.2f} dB")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    run_inference()
