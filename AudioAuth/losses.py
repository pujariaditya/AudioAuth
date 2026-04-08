"""Loss functions for AudioAuth.

Loss modules used across all training stages:

- **Reconstruction:** MultiScaleSTFTLoss, MelSpectrogramLoss, L1Loss
- **Adversarial:** GANLoss (least-squares formulation with feature matching)
- **Watermarking:** LocalizationLoss (BCE-based presence detection),
  DecodingLoss (dual 16-bit watermark extraction)
- **Perceptual:** TFLoudnessLoss (time-frequency loudness difference)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import loudness
from typing import List, Dict, Optional, Tuple, Literal


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for audio quality.

    Computes STFT at multiple window sizes to capture both fine-grained
    transient detail and broad spectral structure.

    Attributes:
        window_lengths: FFT window sizes for each scale.
        overlap: Fractional overlap between windows (used to derive hop sizes).
        eps: Floor value for log-magnitude stability.
        mag_weight: Weight for linear-magnitude L1 term.
        log_weight: Weight for log-magnitude L1 term.
        pow: Exponent applied to magnitudes before loss computation.
        hop_sizes: Derived hop sizes (window_length // 4).
    """
    
    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        overlap: float = 0.75,
        eps: float = 1e-5,
        clamp_eps: Optional[float] = None,  # Alias for eps for compatibility
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0
    ):
        super().__init__()
        
        self.window_lengths = window_lengths
        self.overlap = overlap
        self.eps = clamp_eps if clamp_eps is not None else eps
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.pow = pow
        
        self.hop_sizes = [w // 4 for w in self.window_lengths]
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-scale STFT loss
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            Dictionary with total loss and per-scale losses
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        losses = []
        
        for window_length, hop_size in zip(self.window_lengths, self.hop_sizes):
            pred_stft = torch.stft(
                pred,
                n_fft=window_length,
                hop_length=hop_size,
                window=torch.hann_window(window_length).to(pred.device),
                return_complex=True
            )
            
            target_stft = torch.stft(
                target,
                n_fft=window_length,
                hop_length=hop_size,
                window=torch.hann_window(window_length).to(target.device),
                return_complex=True
            )
            
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            pred_mag_pow = pred_mag.pow(self.pow)
            target_mag_pow = target_mag.pow(self.pow)
            
            mag_loss = F.l1_loss(pred_mag_pow, target_mag_pow)
            
            # Log-magnitude term emphasises perceptually relevant spectral shape
            pred_log_mag = torch.log10(pred_mag_pow + self.eps)
            target_log_mag = torch.log10(target_mag_pow + self.eps)
            log_mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
            
            scale_loss = self.mag_weight * mag_loss + self.log_weight * log_mag_loss
            losses.append(scale_loss)
        
        total_loss = sum(losses) / len(losses)
        
        return {
            "total": total_loss,
            "per_scale": losses
        }


class MelSpectrogramLoss(nn.Module):
    """Multi-scale mel-spectrogram loss for perceptual audio quality.

    Computes L1 in log-mel space at multiple window sizes, weighting
    frequency bands by human auditory sensitivity.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT size (used as default when creating transforms).
        pow: Power exponent for mel spectrograms.
        normalized: Whether mel filterbanks are area-normalised.
        eps: Floor for log stability.
        mag_weight: Optional weight for a linear-magnitude term.
        return_dict: If True, return a dict with per-scale breakdown.
        mel_transforms: ModuleList of torchaudio MelSpectrogram transforms,
            one per scale.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        win_lengths: List[int] = [2048, 512],
        n_mels: List[int] = [150, 80],
        pow: float = 2.0,
        normalized: bool = False,
        eps: float = 1e-5,
        mag_weight: float = 0.0,
        fmin: Optional[List[float]] = None,
        fmax: Optional[List[float]] = None,
        return_dict: bool = False
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.pow = pow
        self.normalized = normalized
        self.eps = eps
        self.mag_weight = mag_weight
        self.return_dict = return_dict
        
        if fmin is None:
            fmin = [0.0] * len(win_lengths)
        if fmax is None:
            fmax = [sample_rate / 2] * len(win_lengths)
        
        self.mel_transforms = nn.ModuleList()
        
        for i, (win_length, n_mel) in enumerate(zip(win_lengths, n_mels)):
            hop_length = win_length // 4
            
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win_length,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mel,
                f_min=fmin[i],
                f_max=fmax[i],
                power=pow,
                normalized=normalized,
                norm="slaney" if normalized else None,
                mel_scale="slaney"
            )
            self.mel_transforms.append(mel_transform)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-scale mel-spectrogram loss
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            Total mel loss or dict if return_dict=True
        """
        loss = 0.0
        losses_per_scale = []
        
        for i, mel_transform in enumerate(self.mel_transforms):
            mel_transform = mel_transform.to(pred.device)

            pred_mel = mel_transform(pred)
            target_mel = mel_transform(target)

            pred_mel = torch.clamp(pred_mel, min=self.eps)
            target_mel = torch.clamp(target_mel, min=self.eps)

            pred_log_mel = torch.log10(pred_mel)
            target_log_mel = torch.log10(target_mel)
            mel_loss = F.l1_loss(pred_log_mel, target_log_mel)

            if self.mag_weight > 0:
                mag_loss = F.l1_loss(pred_mel, target_mel)
                mel_loss = mel_loss + self.mag_weight * mag_loss
            
            loss += mel_loss
            losses_per_scale.append(mel_loss)
        
        total_loss = loss / len(self.mel_transforms)
        
        if self.return_dict:
            return {
                "total": total_loss,
                "per_scale": losses_per_scale
            }
        return total_loss


class GANLoss(nn.Module):
    """Least-squares GAN losses for adversarial training.

    All methods are static; the class carries no learnable state and exists
    for namespacing.  The LSGAN formulation (Mao et al., 2017) avoids the
    vanishing-gradient problem of cross-entropy GAN losses.
    """
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def discriminator_loss(
        real_scores: List[List[torch.Tensor]],
        fake_scores: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute discriminator loss using least-squares formulation
        
        Args:
            real_scores: Discriminator outputs for real audio (list of feature maps)
            fake_scores: Discriminator outputs for fake audio (list of feature maps)
            
        Returns:
            Tuple of (total discriminator loss, dict of individual discriminator losses)
        """
        loss = 0.0
        individual_losses = {}
        
        disc_idx = 0
        for i, (real_out, fake_out) in enumerate(zip(real_scores, fake_scores)):
            if isinstance(real_out[0], list) and isinstance(fake_out[0], list):
                for inner_real, inner_fake in zip(real_out, fake_out):
                    real_score = inner_real[-1]
                    fake_score = inner_fake[-1]
                    
                    real_loss = torch.mean((real_score - 1) ** 2)
                    fake_loss = torch.mean(fake_score ** 2)
                    disc_loss = real_loss + fake_loss
                    
                    loss += disc_loss
                    individual_losses[f'd_{disc_idx}_loss'] = disc_loss.item()
                    disc_idx += 1
            else:
                real_score = real_out[-1]
                fake_score = fake_out[-1]

                real_loss = torch.mean((real_score - 1) ** 2)
                fake_loss = torch.mean(fake_score ** 2)
                disc_loss = real_loss + fake_loss
                
                loss += disc_loss
                individual_losses[f'd_{disc_idx}_loss'] = disc_loss.item()
                disc_idx += 1
        
        total_loss = loss / max(disc_idx, 1)
        individual_losses['d_total_loss'] = total_loss.item()
        
        return total_loss, individual_losses
    
    @staticmethod
    def generator_loss(
        fake_scores: List[List[torch.Tensor]],
        real_scores: Optional[List[List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute generator adversarial loss
        
        Args:
            fake_scores: Discriminator outputs for fake audio (list of feature maps)
            real_scores: Discriminator outputs for real audio (optional, for feature matching)
            
        Returns:
            Tuple of (generator adversarial loss, feature matching loss if real_scores provided)
        """
        loss = 0.0
        count = 0
        
        for fake_out in fake_scores:
            if isinstance(fake_out[0], list):
                for inner_fake_out in fake_out:
                    fake_score = inner_fake_out[-1]
                    loss += torch.mean((fake_score - 1) ** 2)
                    count += 1
            else:
                fake_score = fake_out[-1]
                loss += torch.mean((fake_score - 1) ** 2)
                count += 1
        
        gen_loss = loss / count if count > 0 else loss
        
        feat_loss = None
        if real_scores is not None:
            feat_loss = GANLoss.feature_matching_loss(real_scores, fake_scores)
        
        return gen_loss, feat_loss
    
    @staticmethod
    def feature_matching_loss(
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Compute feature matching loss
        
        Args:
            real_features: List of discriminator features for real audio
            fake_features: List of discriminator features for fake audio
            
        Returns:
            Feature matching loss
        """
        loss = 0.0
        n_feats = 0
        
        for real_out, fake_out in zip(real_features, fake_features):
            if isinstance(real_out[0], list) and isinstance(fake_out[0], list):
                for inner_real, inner_fake in zip(real_out, fake_out):
                    # Skip the final element (discriminator score, not a feature map)
                    for real_feat, fake_feat in zip(inner_real[:-1], inner_fake[:-1]):
                        loss += F.l1_loss(fake_feat, real_feat.detach())
                        n_feats += 1
            else:
                for real_feat, fake_feat in zip(real_out[:-1], fake_out[:-1]):
                    loss += F.l1_loss(fake_feat, real_feat.detach())
                    n_feats += 1
        
        return loss / n_feats if n_feats > 0 else loss


class L1Loss(nn.L1Loss):
    """Thin wrapper around ``nn.L1Loss`` that carries an additional
    ``weight`` attribute for external loss weighting.

    Attributes:
        weight: Multiplicative weight (applied externally by the runner).
    """
    
    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss between predictions and targets
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            L1 loss value
        """
        return super().forward(pred, target)



class LocalizationLoss(nn.Module):
    """BCE loss for single-channel watermark presence detection.

    Trains the locator to predict a per-frame probability of watermark
    presence.  Positive samples (watermarked audio) are supervised against
    the ground-truth mask; optional negative samples (clean audio) are
    supervised against an all-zeros target.

    Attributes:
        pos_weight: Scalar weight for the positive-sample BCE term.
        neg_weight: Scalar weight for the negative-sample BCE term.
        bce_loss: BCEWithLogitsLoss instance (sigmoid applied internally
            for numerical stability).
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, watermarked_locator_output: torch.Tensor, ground_truth_mask: torch.Tensor,
                clean_locator_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute BCE localization loss with optional negative samples.

        Args:
            watermarked_locator_output: [B, 1, T] - single channel logits from locator on watermarked audio
            ground_truth_mask: [B, 1, T] - binary mask (0=no watermark, 1=watermark)
            clean_locator_output: [B, 1, T] - locator output on clean audio (optional)

        Returns:
            Combined localization loss (positive + negative if clean_locator_output provided)
        """
        assert watermarked_locator_output.dim() == 3 and watermarked_locator_output.size(1) == 1, \
            f"Expected watermarked_locator_output shape [B, 1, T], got {watermarked_locator_output.shape}"
        assert ground_truth_mask.dim() == 3 and ground_truth_mask.size(1) == 1, \
            f"Expected ground_truth_mask shape [B, 1, T], got {ground_truth_mask.shape}"

        pos_loss = self.bce_loss(watermarked_locator_output, ground_truth_mask.float())

        if clean_locator_output is not None:
            assert clean_locator_output.dim() == 3 and clean_locator_output.size(1) == 1, \
                f"Expected clean_locator_output shape [B, 1, T], got {clean_locator_output.shape}"

            # Target is all-zeros: clean audio should predict "no watermark"
            clean_target = torch.zeros_like(clean_locator_output)
            neg_loss = self.bce_loss(clean_locator_output, clean_target)

            total_loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss
            return total_loss

        return self.pos_weight * pos_loss


class DecodingLoss(nn.Module):
    """Decoding loss for 32-bit dual watermark extraction.

    Computes BCE separately for the model watermark (bits 0-15) and the
    data watermark (bits 16-31), each weighted independently.  Loss is
    masked to watermarked regions only, so unwatermarked frames do not
    contribute gradients.

    Attributes:
        model_weight: Scalar weight for the model-watermark BCE term.
        data_weight: Scalar weight for the data-watermark BCE term.
        pos_weight: Scalar weight for positive (watermarked) samples.
        neg_weight: Scalar weight for negative (clean) samples.
        bce_loss: Shared BCEWithLogitsLoss instance.
    """
    def __init__(self, model_weight: float = 1.0, data_weight: float = 1.0,
                 pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.model_weight = model_weight
        self.data_weight = data_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, watermarked_detector_output, ground_truth_mask, ground_truth_message,
                clean_detector_output=None):
        """
        Args:
            watermarked_detector_output: Detector output on watermarked audio [B, 32, T]
            ground_truth_mask: Ground truth presence mask [B, 1, T]
                - Single channel indicating watermark presence (applies to both model and data)
            ground_truth_message: Original 32-bit message [B, 32]
            clean_detector_output: Detector output on clean audio [B, 32, T] (optional)

        Returns:
            Combined dual decoding loss
        """
        if ground_truth_message.size(0) == 0:
            return torch.tensor(0.0, device=watermarked_detector_output.device)

        model_logits = watermarked_detector_output[:, :16]  # [B, 16, T]
        data_logits = watermarked_detector_output[:, 16:]   # [B, 16, T]

        model_message = ground_truth_message[:, :16]  # [B, 16]
        data_message = ground_truth_message[:, 16:]   # [B, 16]

        num_samples = model_logits.size(2)

        # Broadcast ground-truth bits across the time dimension
        model_expanded = model_message.unsqueeze(-1).repeat(1, 1, num_samples)  # [B, 16, T]
        data_expanded = data_message.unsqueeze(-1).repeat(1, 1, num_samples)    # [B, 16, T]

        # Restrict loss to watermarked regions only
        mask_bool = ground_truth_mask[:, 0:1, :].bool()  # [B, 1, T]

        if mask_bool.any():
            mask_model = mask_bool.expand_as(model_logits)  # [B, 16, T]
            mask_data = mask_bool.expand_as(data_logits)    # [B, 16, T]

            model_logits_masked = model_logits[mask_model]
            model_target_masked = model_expanded[mask_model]
            model_loss = self.bce_loss(model_logits_masked, model_target_masked.float())

            data_logits_masked = data_logits[mask_data]
            data_target_masked = data_expanded[mask_data]
            data_loss = self.bce_loss(data_logits_masked, data_target_masked.float())

            pos_loss = self.model_weight * model_loss + self.data_weight * data_loss
        else:
            pos_loss = torch.tensor(0.0, device=watermarked_detector_output.device)

        if clean_detector_output is not None:
            clean_model = clean_detector_output[:, :16]  # [B, 16, T]
            clean_data = clean_detector_output[:, 16:]   # [B, 16, T]

            # Clean audio should decode to all-zeros (maximum uncertainty)
            model_neg_loss = self.bce_loss(clean_model, torch.zeros_like(clean_model))
            data_neg_loss = self.bce_loss(clean_data, torch.zeros_like(clean_data))

            neg_loss = self.model_weight * model_neg_loss + self.data_weight * data_neg_loss

            total_loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss
            return total_loss

        # If no negative samples, return only positive loss
        return self.pos_weight * pos_loss


class TFLoudnessLoss(nn.Module):
    """Time-frequency loudness loss.

    Decomposes audio into frequency bands via FFT, windows each band, and
    penalises per-band loudness differences between watermarked and original
    audio.  A softmax weighting focuses the loss on the bands with the
    largest loudness change, encouraging perceptual imperceptibility.

    Attributes:
        num_freq_bands: Number of equal-width frequency bands.
        window_size: Number of samples per analysis window.
        overlap: Overlap in samples between adjacent windows.
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(self, num_freq_bands, window_size, overlap, sample_rate):
        super(TFLoudnessLoss, self).__init__()
        self.num_freq_bands = num_freq_bands
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate

    def _divide_into_bands(self, audio_sample):
        """Split audio into equal-width frequency bands via FFT."""
        fft_length = audio_sample.shape[-1]
        fft_result = torch.fft.fft(audio_sample)
        freq_bins = torch.fft.fftfreq(fft_length, 1 / self.sample_rate)
        band_ranges = torch.linspace(0, self.sample_rate / 2, self.num_freq_bands + 1)
        frequency_bands = torch.zeros(audio_sample.shape[0], self.num_freq_bands, 2, fft_length)

        for i in range(fft_length):
            freq = torch.abs(freq_bins[i])
            band_index = torch.searchsorted(band_ranges, freq) - 1
            
            if band_index < self.num_freq_bands:
                magnitude = torch.abs(fft_result[..., i])
                phase = torch.angle(fft_result[..., i])

                frequency_bands[:, band_index, 0, i] = magnitude.squeeze()
                frequency_bands[:, band_index, 1, i] = phase.squeeze()
                
        return frequency_bands

    def _segment_band_signal(self, signal):
        """Window each frequency-band signal into overlapping chunks."""
        num_samples = signal.shape[-1]
        num_chunks = (num_samples - self.overlap) // (self.window_size - self.overlap)
        chunks = torch.zeros(signal.shape[0], signal.shape[1], num_chunks, self.window_size)

        for i in range(num_chunks):
            start = i * (self.window_size - self.overlap)
            end = start + self.window_size
            chunks[:, :, i, :] = signal[:, :, 0, start:end]

        return chunks

    def _convert_bands_to_segments(self, frequency_bands):
        """Reconstruct time-domain signals from per-band magnitude/phase."""
        batch_size, num_freq_bands, _, fft_length = frequency_bands.shape
        reconstructed_signals = torch.zeros(batch_size, num_freq_bands, 1, fft_length)

        for band_index in range(num_freq_bands):
            magnitudes = frequency_bands[:, band_index, 0]
            phases = frequency_bands[:, band_index, 1]
            complex_bins = magnitudes * torch.exp(1j * phases)
            reconstructed_signal = torch.fft.ifft(complex_bins)
            reconstructed_signals[:, band_index, 0] = reconstructed_signal.real

        return reconstructed_signals

    def forward(self, watermarked_signals, original_signals):
        original_frequency_bands = self._divide_into_bands(original_signals)
        watermarked_frequency_bands = self._divide_into_bands(watermarked_signals)

        reconstructed_original_signals = self._convert_bands_to_segments(original_frequency_bands)
        reconstructed_watermarked_signals = self._convert_bands_to_segments(watermarked_frequency_bands)

        original_chunks = self._segment_band_signal(reconstructed_original_signals)
        watermarked_chunks = self._segment_band_signal(reconstructed_watermarked_signals)

        loudness_differences = []
        for i in range(original_signals.shape[0]):
            for j in range(self.num_freq_bands):
                for k in range(original_chunks.shape[2]):
                    original_sample = original_chunks[i, j, k].unsqueeze(0)
                    watermarked_sample = watermarked_chunks[i, j, k].unsqueeze(0)
                    original_loudness = loudness(original_sample, self.sample_rate)
                    watermarked_loudness = loudness(watermarked_sample, self.sample_rate)
                    if torch.isnan(original_loudness):
                        original_loudness = torch.tensor(0.0)
                    if torch.isnan(watermarked_loudness):
                        watermarked_loudness = torch.tensor(0.0)
                    loudness_difference = watermarked_loudness - original_loudness
                    loudness_differences.append(loudness_difference)

        tf_loudness_diff_values = torch.tensor(loudness_differences)

        assert not torch.isnan(tf_loudness_diff_values).any(), "tf_loudness_diff_values contains NaN values"

        # Softmax weighting focuses the penalty on bands with the largest
        # loudness deviation, prioritising perceptually salient differences.
        softmax_weights = F.softmax(tf_loudness_diff_values, dim=0)

        assert not torch.isnan(softmax_weights).any(), "softmax_weights contains NaN values"

        tf_loudness_loss = (softmax_weights * tf_loudness_diff_values).sum()

        assert not torch.isnan(tf_loudness_loss), "tf_loudness_loss is NaN"

        return tf_loudness_loss
