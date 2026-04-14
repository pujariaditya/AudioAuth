<p align="center">
  <img src="figures/AudioAuth_TeaserImage.png" width="90%" alt="AudioAuth Overview"/>
</p>

<h1 align="center">AudioAuth</h1>

<p align="center">
  A dual-watermarking framework for robust audio integrity verification and source attribution.
</p>

<p align="center">
  <a href="https://www.researchgate.net/publication/403305808_AudioAuth_A_Dual-Watermarking_Framework_for_Robust_Audio_Integrity_and_Source_Attribution"><img src="https://img.shields.io/badge/Paper-ResearchGate-00CCBB.svg" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg" alt="PyTorch 2.2+">
</p>

---

AudioAuth embeds two complementary watermarks into audio using frequency-partitioned encoding: a **model watermark** (fixed pattern in even frequency bands) for source attribution, and a **data watermark** (dynamic payload in odd frequency bands) for fine-grained temporal tamper localization.

## Installation

```bash
git clone https://github.com/pujariaditya/AudioAuth.git
cd AudioAuth
pip install -e .
```

## Quick Start

```python
from pathlib import Path
import torch
from AudioAuth import Config, load_models
from AudioAuth.processor import load_audio

# Load config and model
config = Config.from_sources(yaml_file=Path("configs/train_stage3.yml"))
model, _ = load_models(config, checkpoint_path=Path("checkpoints/best.pth"), device="cuda")
model.eval()

# Load audio and create watermark message
audio = load_audio("input.wav", sample_rate=16000).unsqueeze(0).unsqueeze(0).to("cuda")  # [1, 1, T]
# Create watermark message (16 model bits + 16 data bits)
model_bits = torch.ones(1, 16, device="cuda")                    # fixed model signature
data_bits  = torch.randint(0, 2, (1, 16), device="cuda").float() # dynamic payload
msg = torch.cat([model_bits, data_bits], dim=1)                   # [1, 32]

# Embed watermark
with torch.no_grad():
    out = model(audio, msg, phase="audio_sample")

watermarked = out["watermarked_signal"]       # [1, 1, T] — watermarked audio
detected    = out["watermarked_detector_output"]  # [1, 32]  — 16 model + 16 data bits
locator     = out["watermarked_locator_output"]   # [1, 1, T] — per-sample watermark presence
```

## Training

AudioAuth uses a 3-stage training pipeline:

```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py --audio-dir /path/to/audio --output-dir data/

# 2. Stage 1: Base watermarking
python train.py --cfg-path configs/train_stage1.yml

# 3. Stage 2: Attack robustness
python train.py --cfg-path configs/train_stage2.yml

# 4. Stage 3: GAN fine-tuning
python train.py --cfg-path configs/train_stage3.yml
```

For multi-GPU training, use `torchrun --nproc_per_node=N train.py --cfg-path ...`. Set `resume_from` in each config to the previous stage's checkpoint.

## Audio Samples

Compare original and watermarked audio — the watermark is imperceptible.

| Sample | Original | Watermarked | SI-SNR (dB) |
|--------|----------|-------------|-------------|
| sample_001 | [Play](demo/samples/original/sample_001_original.wav) | [Play](demo/samples/watermarked/sample_001_watermarked.wav) | 22.46 |
| sample_002 | [Play](demo/samples/original/sample_002_original.wav) | [Play](demo/samples/watermarked/sample_002_watermarked.wav) | 21.50 |
| sample_003 | [Play](demo/samples/original/sample_003_original.wav) | [Play](demo/samples/watermarked/sample_003_watermarked.wav) | 20.00 |

## Citation

```bibtex
@article{pujari2026audioauth,
  author    = {Pujari, Aditya and Rattani, Ajita},
  title     = {AudioAuth: A Dual-Watermarking Framework for Robust Audio Integrity and Source Attribution},
  journal   = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year      = {2026},
  month     = {01},
  pages     = {1-15},
  doi       = {10.1109/TBIOM.2026.3679274}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
