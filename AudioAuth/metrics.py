"""Audio watermarking and reconstruction evaluation metrics.

Provides evaluation metrics for:
- Audio reconstruction quality (SI-SNR, PESQ, STOI)
- Watermark detection accuracy (BER - Bit Error Rate)
- Watermark localization accuracy (mIoU - Mean Intersection over Union)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

try:
    from pesq import pesq, NoUtterancesError
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    warnings.warn("PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    warnings.warn("STOI not available. Install with: pip install pystoi")


class SISNR(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) metric.
    
    The computation follows the standard SI-SNR formula:
    1. Zero-mean both signals
    2. Compute optimal scaling factor (projection)
    3. Calculate SI-SNR as 10*log10(signal_power/noise_power)
    
    Args:
        epsilon: Small value to prevent division by zero (default: 1e-8)
        scaling: If True, compute SI-SNR with optimal scaling (default).
                If False, compute regular SNR without scaling.
    """
    def __init__(self, epsilon=1e-8, scaling=True):
        super().__init__()
        self.epsilon = epsilon
        self.scaling = scaling
    
    def forward(self, estimates, references):
        # Ensure equal length
        min_len = min(references.shape[-1], estimates.shape[-1])
        references = references[..., :min_len]
        estimates = estimates[..., :min_len]
        
        # Flatten batch and channel dimensions
        ref_sig = references.reshape(-1, references.shape[-1])
        out_sig = estimates.reshape(-1, estimates.shape[-1])
        
        # Zero mean
        ref_sig = ref_sig - torch.mean(ref_sig, dim=-1, keepdim=True)
        out_sig = out_sig - torch.mean(out_sig, dim=-1, keepdim=True)
        
        if self.scaling:
            # Compute optimal scaling factor for SI-SNR
            ref_energy = torch.sum(ref_sig ** 2, dim=-1, keepdim=True) + self.epsilon
            proj = torch.sum(ref_sig * out_sig, dim=-1, keepdim=True) * ref_sig / ref_energy
        else:
            # No scaling for regular SNR
            proj = ref_sig
        
        # Compute SI-SNR/SNR
        noise = out_sig - proj
        ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + self.epsilon)
        si_snr = 10 * torch.log10(ratio + self.epsilon)
        
        # Return SI-SNR as a metric
        return torch.mean(si_snr)



class PESQ(nn.Module):
    def __init__(self, sample_rate=16000, mode='wb'):
        super().__init__()
        if not HAS_PESQ:
            raise ImportError("PESQ not available. Install with: pip install pesq")
        
        if sample_rate not in [8000, 16000]:
            raise ValueError("PESQ only supports 8kHz or 16kHz")
        
        self.sample_rate = sample_rate
        self.mode = mode
    
    def forward(self, estimates, references):
        # Convert to numpy
        references_np = references.squeeze(1).cpu().numpy()
        estimates_np = estimates.squeeze(1).cpu().numpy()

        scores = []
        for i in range(references_np.shape[0]):
            try:
                score = pesq(self.sample_rate, references_np[i], estimates_np[i], self.mode)
                scores.append(score)
            except NoUtterancesError as e:
                # Skip samples with no utterances detected
                continue
        
        return torch.tensor(np.mean(scores), device=references.device)


class STOI(nn.Module):
    def __init__(self, sample_rate=16000, extended=False):
        super().__init__()
        if not HAS_STOI:
            raise ImportError("STOI not available. Install with: pip install pystoi")
        
        self.sample_rate = sample_rate
        self.extended = extended
    
    def forward(self, estimates, references):
        # Convert to numpy
        references_np = references.squeeze(1).cpu().numpy()
        estimates_np = estimates.squeeze(1).cpu().numpy()
        
        scores = []
        for i in range(references_np.shape[0]):
            score = stoi(references_np[i], estimates_np[i], self.sample_rate, extended=self.extended)
            scores.append(score)
        
        return torch.tensor(np.mean(scores), device=references.device)


# === Watermark Detection Metrics ===

def _compute_accuracy(positive, negative):
    """Compute detection accuracy for locator outputs.

    Adapted for single-channel locator output.

    Args:
        positive: [B, 1, T] locator outputs on watermarked audio (logits)
        negative: [B, 1, T] locator outputs on clean audio (logits)

    Returns:
        Accuracy as float
    """
    # Convert logits to probabilities
    pos_probs = torch.sigmoid(positive[:, 0, :])  # [B, T]
    neg_probs = torch.sigmoid(negative[:, 0, :])  # [B, T]

    # Count correct predictions
    # Positive: watermark correctly detected (prob > 0.5)
    # Negative: no watermark correctly identified (prob <= 0.5)
    N = (pos_probs.mean(dim=1) > 0.5).sum() + \
        (neg_probs.mean(dim=1) <= 0.5).sum()
    acc = N.float() / (2 * positive.size(0))
    return acc.item()


def _compute_FPR(negative):
    """Compute false positive rate.

    Adapted for single-channel locator output.

    Args:
        negative: [B, 1, T] locator outputs on clean audio (logits)

    Returns:
        False positive rate as float
    """
    # Convert logits to probabilities
    neg_probs = torch.sigmoid(negative[:, 0, :])  # [B, T]

    # False positive: clean audio predicted as watermarked (prob > 0.5)
    N = (neg_probs.mean(dim=1) > 0.5).sum()
    fpr = N.float() / negative.size(0)
    return fpr.item()


def _compute_FNR(positive):
    """Compute false negative rate.

    Adapted for single-channel locator output.

    Args:
        positive: [B, 1, T] locator outputs on watermarked audio (logits)

    Returns:
        False negative rate as float
    """
    # Convert logits to probabilities
    pos_probs = torch.sigmoid(positive[:, 0, :])  # [B, T]

    # False negative: watermarked audio predicted as clean (prob <= 0.5)
    N = (pos_probs.mean(dim=1) <= 0.5).sum()
    fnr = N.float() / positive.size(0)
    return fnr.item()


def _compute_bit_acc(positive, original, mask=None):
    """Compute bit accuracy from detector outputs.

    Bit accuracy computation for the separate detector.

    Args:
        positive: [B, nbits, T] detector outputs (raw logits from WVerify's separate detector)
        original: [B, nbits] original message (0 or 1)
        mask: Optional [B, 1, T] mask for partial watermarking

    Returns:
        Bit accuracy as float
    """
    decoded = positive  # No need to extract from channel 2+ since detector is separate

    if mask is not None:
        # Cut last dim of positive to keep only where mask is 1
        new_shape = [*decoded.shape[:-1], -1]  # [B, nbits, -1]
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)

    # Apply sigmoid to convert logits to probabilities, average over time, then threshold at 0.5
    decoded = torch.sigmoid(decoded).mean(dim=-1) > 0.5  # [B, nbits]

    # Compute accuracy
    bit_acc = (decoded == original).float().mean()
    return bit_acc.item()


def _compute_localized_ber(detector_output, original, mask):
    """Compute localized Bit Error Rate only in watermarked regions.

    This function explicitly computes BER only where the watermark is present
    (mask == 1), providing clear metrics for watermarked segments.

    Args:
        detector_output: [B, nbits, T] detector outputs (raw logits)
        original: [B, nbits] original message bits (0 or 1)
        mask: [B, 1, T] binary mask (1 = watermark present, 0 = no watermark)

    Returns:
        Tuple of (localized_ber, num_bits_evaluated, per_bit_ber)
        - localized_ber: BER computed only in masked regions (float)
        - num_bits_evaluated: Total number of bits evaluated (int)
        - per_bit_ber: [B, nbits] BER per bit across time (tensor)
    """
    if mask is None:
        raise ValueError("Mask is required for localized BER computation")

    nbits, T = detector_output.shape[1], detector_output.shape[2]

    # Expand original bits to match temporal dimension
    original_expanded = original.unsqueeze(-1).expand(-1, -1, T)  # [B, nbits, T]

    # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
    detector_preds = (torch.sigmoid(detector_output) > 0.5).float()  # [B, nbits, T]

    # Compute bit errors only where mask is 1
    # Expand mask from [B, 1, T] to [B, nbits, T]
    mask_expanded = mask.expand(-1, nbits, -1)

    # Bit errors in masked regions
    bit_errors = (detector_preds != original_expanded.float()) * mask_expanded.float()  # [B, nbits, T]

    # Count total errors and total bits in masked regions
    total_errors = bit_errors.sum()
    total_bits = mask_expanded.sum()  # Total bit positions where mask is 1

    # Compute localized BER
    localized_ber = (total_errors / (total_bits + 1e-8)).item()

    # Compute per-bit BER across time (for analysis)
    # Sum errors and mask counts per bit
    errors_per_bit = bit_errors.sum(dim=-1)  # [B, nbits]
    mask_counts_per_bit = mask_expanded.sum(dim=-1)  # [B, nbits]
    per_bit_ber = errors_per_bit / (mask_counts_per_bit + 1e-8)  # [B, nbits]

    return localized_ber, int(total_bits.item()), per_bit_ber


def _compute_miou(pred_mask, target_mask, threshold=0.5):
    """Compute Mean IoU for temporal segmentation.
    
    Args:
        pred_mask: [B, 1, T] predicted mask (probabilities) - single channel
        target_mask: [B, 1, T] ground truth mask (binary) - single channel
        threshold: Threshold for binarizing predictions
    
    Returns:
        Mean IoU as float
    """
    # Ensure 2D
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(1)  # [B, 1, T] -> [B, T]
    if target_mask.dim() == 3:
        target_mask = target_mask.squeeze(1)  # [B, 1, T] -> [B, T]
    
    # Binarize predictions
    pred_binary = (pred_mask > threshold).float()
    target_binary = target_mask.float()
    
    # Compute IoU per sample
    intersection = (pred_binary * target_binary).sum(dim=1)
    union = ((pred_binary + target_binary) > 0).float().sum(dim=1)
    
    # Handle empty masks (when union is 0, both masks are empty, IoU should be 1)
    epsilon = 1e-7
    iou = torch.where(
        union > 0,
        intersection / (union + epsilon),
        torch.ones_like(union)
    )
    
    return iou.mean().item()


def compute_localization_metrics(positive, negative,
                                target_mask_pos=None, target_mask_neg=None,
                                threshold=0.5):
    """Compute all localization metrics in a bundled function.

    Supports both single-channel and dual-channel outputs.

    Args:
        positive: [B, 1, T] or [B, 2, T] locator outputs on watermarked audio (logits)
        negative: [B, 1, T] or [B, 2, T] locator outputs on clean audio (logits)
        target_mask_pos: Optional [B, 1, T] or [B, 2, T] ground truth mask for positive samples
        target_mask_neg: Optional [B, 1, T] or [B, 2, T] ground truth mask for negative samples
        threshold: Threshold for mIoU computation

    Returns:
        Dictionary with localization metrics:
        For single-channel: loc_acc, loc_fpr, loc_fnr, loc_tpr, miou_pos, miou_neg, miou
        For dual-channel: model_loc_acc, data_loc_acc, model_loc_fpr, data_loc_fpr,
                         model_loc_fnr, data_loc_fnr, model_loc_tpr, data_loc_tpr,
                         model_miou, data_miou, combined_miou
    """
    metrics = {}

    # Check if single or dual channel
    is_dual_channel = positive.shape[1] == 2

    if is_dual_channel:
        # Dual-channel handling (original functionality)
        assert negative.shape[1] == 2, f"Expected 2 channels, got {negative.shape[1]}"

        # Split channels
        positive_model = positive[:, 0:1, :]  # [B, 1, T]
        positive_data = positive[:, 1:2, :]   # [B, 1, T]
        negative_model = negative[:, 0:1, :]  # [B, 1, T]
        negative_data = negative[:, 1:2, :]   # [B, 1, T]

        # Model watermark metrics
        metrics['model_loc_acc'] = _compute_accuracy(positive_model, negative_model)
        metrics['model_loc_fpr'] = _compute_FPR(negative_model)
        metrics['model_loc_fnr'] = _compute_FNR(positive_model)
        metrics['model_loc_tpr'] = 1.0 - metrics['model_loc_fnr']

        # Data watermark metrics
        metrics['data_loc_acc'] = _compute_accuracy(positive_data, negative_data)
        metrics['data_loc_fpr'] = _compute_FPR(negative_data)
        metrics['data_loc_fnr'] = _compute_FNR(positive_data)
        metrics['data_loc_tpr'] = 1.0 - metrics['data_loc_fnr']

        # mIoU computation if masks are provided
        if target_mask_pos is not None:
            # Verify mask has 2 channels
            assert target_mask_pos.shape[1] == 2, f"Expected 2-channel mask, got {target_mask_pos.shape}"

            # Model channel IoU - use channel 0 of mask
            pred_mask_model = torch.sigmoid(positive_model)
            metrics['model_miou'] = _compute_miou(pred_mask_model, target_mask_pos[:, 0:1, :], threshold)

            # Data channel IoU - use channel 1 of mask
            pred_mask_data = torch.sigmoid(positive_data)
            metrics['data_miou'] = _compute_miou(pred_mask_data, target_mask_pos[:, 1:2, :], threshold)

            # Combined IoU (both watermarks must be detected)
            combined_pred = (torch.sigmoid(positive_model) > threshold) & (torch.sigmoid(positive_data) > threshold)
            combined_target = (target_mask_pos[:, 0:1, :] > 0.5) & (target_mask_pos[:, 1:2, :] > 0.5)
            metrics['combined_miou'] = _compute_miou(combined_pred.float(), combined_target.float(), threshold=0.5)
    else:
        # Single-channel handling (from mm.py)
        # Core localization metrics - use loc_ prefix for consistency with config
        metrics['loc_acc'] = _compute_accuracy(positive, negative)
        metrics['loc_fpr'] = _compute_FPR(negative)
        metrics['loc_fnr'] = _compute_FNR(positive)
        metrics['loc_tpr'] = 1.0 - metrics['loc_fnr']

        # Temporal segmentation metrics (if masks provided)
        if target_mask_pos is not None:
            # Single channel output - convert logits to probabilities
            pred_mask_pos = torch.sigmoid(positive)  # [B, 1, T]
            metrics['miou_pos'] = _compute_miou(pred_mask_pos, target_mask_pos, threshold)

        if target_mask_neg is not None:
            # Single channel output - convert logits to probabilities
            pred_mask_neg = torch.sigmoid(negative)  # [B, 1, T]
            metrics['miou_neg'] = _compute_miou(pred_mask_neg, target_mask_neg, threshold)

        # Average mIoU if both are available
        if 'miou_pos' in metrics and 'miou_neg' in metrics:
            metrics['miou'] = (metrics['miou_pos'] + metrics['miou_neg']) / 2.0

    return metrics



def compute_detection_metrics(positive, negative, original, mask=None):
    """Compute detection metrics for dual watermarking (model and data).

    Always expects 32-bit dual watermarking with separate model and data watermarks.

    Args:
        positive: [B, 32, T] detector outputs on watermarked audio (raw logits)
                 Bits 0-15: Model watermark, Bits 16-31: Data watermark
        negative: [B, 32, T] detector outputs on clean audio (raw logits)
        original: [B, 32] original message bits (0 or 1)
                 Bits 0-15: Model watermark, Bits 16-31: Data watermark
        mask: Optional [B, 1, T] mask for partial watermarking
              (1 = watermark present, 0 = no watermark)

    Returns:
        Dictionary with dual watermark detection metrics:
        - model_ber: Model watermark BER
        - data_ber: Data watermark BER
        - model_localized_ber: Model watermark BER in watermarked regions
        - data_localized_ber: Data watermark BER in watermarked regions
        Plus localized metrics when mask is provided
    """
    metrics = {}

    # Verify we have 32-bit dual watermarking
    assert positive.shape[1] == 32, f"Expected 32 bits, got {positive.shape[1]}"
    assert negative.shape[1] == 32, f"Expected 32 bits, got {negative.shape[1]}"
    assert original.shape[1] == 32, f"Expected 32 bits, got {original.shape[1]}"

    # Split 32-bit outputs and messages into model and data watermarks
    positive_model = positive[:, :16, :]  # [B, 16, T] - Model watermark
    positive_data = positive[:, 16:, :]   # [B, 16, T] - Data watermark
    negative_model = negative[:, :16, :]  # [B, 16, T]
    negative_data = negative[:, 16:, :]   # [B, 16, T]

    original_model = original[:, :16]  # [B, 16] - Model bits
    original_data = original[:, 16:]   # [B, 16] - Data bits

    # Compute global metrics for model watermark
    model_bit_acc = _compute_bit_acc(positive_model, original_model, mask=None)
    zeros_model = torch.zeros_like(original_model)
    model_bit_acc_neg = _compute_bit_acc(negative_model, zeros_model, mask=None)

    metrics['model_ber'] = 1.0 - model_bit_acc
    metrics['model_ber_neg'] = 1.0 - model_bit_acc_neg

    # Compute global metrics for data watermark
    data_bit_acc = _compute_bit_acc(positive_data, original_data, mask=None)
    zeros_data = torch.zeros_like(original_data)
    data_bit_acc_neg = _compute_bit_acc(negative_data, zeros_data, mask=None)

    metrics['data_ber'] = 1.0 - data_bit_acc
    metrics['data_ber_neg'] = 1.0 - data_bit_acc_neg

    # Compute localized metrics if mask is provided
    if mask is not None:
        # Use mask for both model and data watermarks
        # Localized metrics for model watermark
        model_loc_ber, model_num_bits, model_per_bit_ber = _compute_localized_ber(
            positive_model, original_model, mask
        )
        metrics['model_localized_ber'] = model_loc_ber

        # Localized metrics for data watermark - use same mask
        data_loc_ber, data_num_bits, data_per_bit_ber = _compute_localized_ber(
            positive_data, original_data, mask
        )
        metrics['data_localized_ber'] = data_loc_ber

        # Localized metrics for negative samples (clean audio)
        # Model watermark negative
        model_loc_ber_neg, _, model_per_bit_ber_neg = _compute_localized_ber(
            negative_model, zeros_model, mask
        )
        metrics['model_localized_ber_neg'] = model_loc_ber_neg

        # Data watermark negative
        data_loc_ber_neg, _, data_per_bit_ber_neg = _compute_localized_ber(
            negative_data, zeros_data, mask
        )
        metrics['data_localized_ber_neg'] = data_loc_ber_neg
    else:
        # When no mask provided, use global metrics as localized
        metrics['model_localized_ber'] = metrics['model_ber']
        metrics['data_localized_ber'] = metrics['data_ber']

    return metrics


# Public aliases
compute_accuracy = _compute_accuracy
compute_FPR = _compute_FPR
compute_FNR = _compute_FNR
compute_miou = _compute_miou
compute_bit_acc = _compute_bit_acc
compute_localized_ber = _compute_localized_ber


class AudioReconstructionMetrics(nn.Module):
    """Audio reconstruction quality metrics.
    
    Evaluates audio quality using perceptual and objective metrics.
    """
    
    def __init__(self,
                 sample_rate=16000,
                 use_pesq=True,
                 use_stoi=True,
                 pesq_sample_rate=16000,
                 pesq_mode='wb'):
        super().__init__()
        
        self.sample_rate = sample_rate
        
        # Basic metrics
        self.sisnr = SISNR()
        
        # Perceptual metrics
        self.use_pesq = use_pesq and HAS_PESQ
        self.use_stoi = use_stoi and HAS_STOI
        
        if self.use_pesq:
            # PESQ requires 8kHz or 16kHz
            if pesq_sample_rate not in [8000, 16000]:
                warnings.warn(f"PESQ sample rate {pesq_sample_rate} not supported, using 16000 Hz")
                pesq_sample_rate = 16000
            self.pesq = PESQ(sample_rate=pesq_sample_rate, mode=pesq_mode)
            self.pesq_sample_rate = pesq_sample_rate
        
        if self.use_stoi:
            self.stoi = STOI(sample_rate=sample_rate)
    
    def forward(self, y_pred, y_true):
        metrics = {}
        
        # Basic metrics
        metrics['sisnr'] = self.sisnr(y_pred, y_true)
        
        # Perceptual metrics
        if self.use_pesq:
            # Resample if needed
            if self.sample_rate != self.pesq_sample_rate:
                import torchaudio.functional as AF
                y_true_pesq = AF.resample(y_true, self.sample_rate, self.pesq_sample_rate)
                y_pred_pesq = AF.resample(y_pred, self.sample_rate, self.pesq_sample_rate)
            else:
                y_true_pesq = y_true
                y_pred_pesq = y_pred
            
            # PESQ expects mono
            if y_true_pesq.shape[1] > 1:
                y_true_pesq = torch.mean(y_true_pesq, dim=1, keepdim=True)
                y_pred_pesq = torch.mean(y_pred_pesq, dim=1, keepdim=True)
            
            metrics['pesq'] = self.pesq(y_pred_pesq, y_true_pesq)
        
        if self.use_stoi:
            # STOI expects mono
            y_true_stoi = y_true
            y_pred_stoi = y_pred
            if y_true_stoi.shape[1] > 1:
                y_true_stoi = torch.mean(y_true_stoi, dim=1, keepdim=True)
                y_pred_stoi = torch.mean(y_pred_stoi, dim=1, keepdim=True)

            metrics['stoi'] = self.stoi(y_pred_stoi, y_true_stoi)
        
        return metrics


def compute_audio_reconstruction_metrics(y_pred, y_true,
                                        sample_rate=16000,
                                        metrics_list=None,
                                        pesq_sample_rate=16000,
                                        pesq_mode='wb'):
    """Compute audio reconstruction quality metrics.
    
    Args:
        y_pred: Predicted/reconstructed audio (watermarked audio)
        y_true: Ground truth/reference audio (original audio)
        sample_rate: Audio sample rate
        metrics_list: List of metrics to compute
        pesq_sample_rate: Sample rate for PESQ (8000 or 16000)
        pesq_mode: PESQ mode ('wb' for wideband, 'nb' for narrowband)
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics_list is None:
        metrics_list = ['sisnr', 'pesq', 'stoi']
    
    # Initialize AudioReconstructionMetrics based on requested metrics
    audio_metrics = AudioReconstructionMetrics(
        sample_rate=sample_rate,
        use_pesq='pesq' in metrics_list,
        use_stoi='stoi' in metrics_list,
        pesq_sample_rate=pesq_sample_rate,
        pesq_mode=pesq_mode
    )
    
    with torch.no_grad():
        all_metrics = audio_metrics(y_pred, y_true)
    
    # Filter to only requested metrics
    result = {}
    for metric in metrics_list:
        if metric in all_metrics:
            result[metric] = all_metrics[metric].item() if torch.is_tensor(all_metrics[metric]) else all_metrics[metric]
    
    return result


