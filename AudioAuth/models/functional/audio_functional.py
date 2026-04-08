"""
Audio signal processing functions.

Provides functional implementations of STFT, ISTFT, MDCT, STDCT, and mel spectrogram
operations for use in audio watermarking pipelines.
"""

import warnings
from typing import Optional, Dict
import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


mel_basis: Dict[str, Tensor] = {}
hann_window: Dict[str, Tensor]  = {}
mdct_filter: Dict[str, Tensor]  = {}
dct_filter: Dict[str, Tensor]  = {}
dct_window_square: Dict[str, Tensor]  = {}


def get_dct_filter(y: Tensor, N: int, win_size: Optional[int], win_type: int,
                   window: Optional[Tensor]) -> Tensor:
    """Build (and cache) the windowed DCT-II filter bank.

    Args:
        y: Reference tensor used to infer dtype and device.
        N: DCT size (number of frequency bins).
        win_size: Window length; defaults to *N* if ``None``.
        win_type: Window function name (e.g., ``'hann'``), or ``None``
            for a rectangular window.
        window: Pre-computed window tensor.  Overrides *win_type* when given.

    Returns:
        Filter tensor of shape ``(N, 1, N)``.
    """
    global dct_filter
    N_win_dtype_device = f"{N}_{win_size}_{y.dtype}_{y.device}"
    if N_win_dtype_device not in dct_filter:
        global dct_window_square
        if win_size is None:
            win_size = N
        if window is not None:
            win_size = window.size(-1)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        elif win_type is None:
            window = torch.ones(N, dtype=torch.float32, device=y.device)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=y.device)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        assert N >= win_size, f"N({N}) must be bigger than win_size({win_size})"
        n = torch.arange(N, dtype=torch.float32, device=y.device).view(1, 1, N)
        k = n.view(N, 1, 1)
        _filter = torch.cos(math.pi/N*k*(n+0.5)) * math.sqrt(2/N)
        _filter[0, 0, :] /= math.sqrt(2)
        dct_filter[N_win_dtype_device] = (_filter * window.view(1, 1, N)).to(y.dtype)
        dct_window_square[N_win_dtype_device] = window.square()
    return dct_filter[N_win_dtype_device]


def stdct(y: Tensor, N: int, hop_size: int, win_size: Optional[int] = None,
          center: bool = False, win_type: str = "hann",
          window: Optional[Tensor] = None) -> Tensor:
    """Short-Time Discrete Cosine Transform II.

    Args:
        y: Input waveform ``[B, 1, hop_size*T]`` or ``[B, hop_size*T]``.
        N: DCT size.
        hop_size: Hop length in samples.
        win_size: Window length; defaults to *N*.
        center: If ``True``, center-pad the input (adds one extra frame).
        win_type: Window function name.
        window: Optional pre-computed window tensor.

    Returns:
        DCT coefficients ``[B, N, T(+1)]``.
    """
    if y.dim() == 2:
        y = y.unsqueeze(1)
    
    _filter = get_dct_filter(y, N, win_size, win_type, window)
    padding = N // 2 if center else (N - hop_size) // 2
    return F.conv1d(y, _filter, bias=None, stride=hop_size, padding=padding)


def istdct(y: Tensor, N: int, hop_size: int, win_size: Optional[int] = None,
           center: bool = False, win_type: str = "hann",
           window: Optional[Tensor] = None) -> Tensor:
    """Inverse Short-Time Discrete Cosine Transform II.

    Args:
        y: DCT coefficients ``[B, N, T(+1)]``.
        N: DCT size.
        hop_size: Hop length in samples.
        win_size: Window length; defaults to *N*.
        center: Must match the *center* flag used in :func:`stdct`.
        win_type: Window function name.
        window: Optional pre-computed window tensor.

    Returns:
        Reconstructed waveform ``[B, 1, hop_size*T]``.
    """
    global dct_window_square
    
    _filter = get_dct_filter(y, N, win_size, win_type, window)
    padding = N // 2 if center else (N - hop_size) // 2
    signal = F.conv_transpose1d(y, _filter, bias=None, stride=hop_size, padding=padding)

    N_win_dtype_device = f"{N}_{win_size}_{y.dtype}_{y.device}"
    window_square = dct_window_square[N_win_dtype_device].view(1, -1, 1).expand(
        y.size(0), -1, y.size(-1))
    window_square_inverse = F.fold(
        window_square,
        output_size = (1, hop_size*y.size(-1) + (N-hop_size) - 2*padding),
        kernel_size = (1, N),
        stride = (1, hop_size),
        padding = (0, padding)
    ).squeeze(2)

    # NOLA(Nonzero Overlap-add) constraint
    assert torch.all(torch.ne(window_square_inverse, 0.0))
    return signal / window_square_inverse


def mdct(y: Tensor, N: int, normalize: int = False) -> Tensor:
    """Modified Discrete Cosine Transform.

    Args:
        y: Input waveform ``[B, 1, N*T]``.
        N: Transform size (window is ``2*N``).
        normalize: If truthy, scale the filter by ``1/sqrt(N)``.

    Returns:
        MDCT coefficients ``[B, N, T+1]``.
    """
    global mdct_filter
    N_dtype_device = f"{N}_{y.dtype}_{y.device}"
    if N_dtype_device not in mdct_filter:
        k = torch.arange(N, dtype=torch.float32, device=y.device).view(N, 1, 1)
        n = torch.arange(2*N, dtype=torch.float32, device=y.device).view(1, 1, 2*N)
        mdct_filter[N_dtype_device] = torch.cos(math.pi/N*(n+0.5+N/2)*(k+0.5))
    _filter = mdct_filter[N_dtype_device]
    if normalize:
        _filter = _filter / math.sqrt(N)
    return F.conv1d(y, _filter, bias=None, stride=N, padding=N)


def imdct(y: Tensor, N: int, normalize: bool = False) -> Tensor:
    """Inverse Modified Discrete Cosine Transform.

    Args:
        y: MDCT coefficients ``[B, N, T+1]``.
        N: Transform size.
        normalize: Must match the flag used in :func:`mdct`.

    Returns:
        Reconstructed waveform ``[B, 1, N*T]``.
    """
    global mdct_filter
    N_dtype_device = f"{N}_{y.dtype}_{y.device}"
    if N_dtype_device not in mdct_filter:
        k = torch.arange(N, dtype=torch.float32, device=y.device).view(N, 1, 1)
        n = torch.arange(2*N, dtype=torch.float32, device=y.device).view(1, 1, 2*N)
        mdct_filter[N_dtype_device] = torch.cos(math.pi/N*(n+0.5+N/2)*(k+0.5))
    _filter = mdct_filter[N_dtype_device]
    if normalize:
        _filter = _filter / math.sqrt(N)
    else:
        _filter = _filter / N
    return F.conv_transpose1d(y, _filter, bias=None, stride=N, padding=N)


def stft_new(y: Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False,
             magnitude: bool = True) -> Tensor:
    """Compute STFT using ``return_complex=True`` and ``torch.view_as_real``.

    Args:
        y: Input waveform.
        n_fft: FFT size.
        hop_size: Hop length.
        win_size: Window length.
        center: Whether to center-pad.
        magnitude: If ``True``, return magnitude spectrum; otherwise real+imag.

    Returns:
        Magnitude or complex-valued spectrogram tensor.
    """
    if torch.min(y) < -1.:
        warnings.warn(f'stft_new: min value is {torch.min(y).item():.4f}')
    if torch.max(y) > 1.:
        warnings.warn(f'stft_new: max value is {torch.max(y).item():.4f}')

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device)

    y = F.pad(y.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(0)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window[wnsize_dtype_device], center=center, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True)
    
    if magnitude:
        spec = torch.view_as_real(spec)
        mag = torch.linalg.norm(spec, dim=-1)
        return mag
    else:
        return torch.view_as_real(spec)


def stft(y: Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False,
         magnitude: bool = True, check_value: bool = False, normalized: bool = False) -> Tensor:
    """Compute STFT with cached Hann window and optional reflect padding.

    Args:
        y: Input waveform ``[B, T]`` or ``[B, 1, T]``.
        n_fft: FFT size.
        hop_size: Hop length.
        win_size: Window length.
        center: Whether to center-pad the input.
        magnitude: If ``True``, return magnitude; otherwise real+imag ``[..., 2]``.
        check_value: Warn if values fall outside [-1, 1].
        normalized: Whether to use normalized STFT.

    Returns:
        Spectrogram tensor.
    """
    if y.dim() == 3:  # [B, 1, T] -> [B, T]
        y = y.squeeze(1)
    
    if check_value:
        if torch.min(y) < -1.:
            warnings.warn(f'stft: min value is {torch.min(y).item():.4f}')
        if torch.max(y) > 1.:
            warnings.warn(f'stft: max value is {torch.max(y).item():.4f}')

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device)

    if not center:
        y = F.pad(
            y.unsqueeze(0),
            (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)),
            mode='reflect'
        )
        y = y.squeeze(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
            window=hann_window[wnsize_dtype_device], center=center, pad_mode='reflect',
            normalized=normalized, onesided=True, return_complex=False)
    
    if magnitude:
        spec = torch.linalg.norm(spec, dim=-1)

    return spec


def istft(spec: Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False,
          normalized: bool = False) -> Tensor:
    """Inverse STFT. Currently only supports ``center=True``.

    Args:
        spec: Complex spectrogram ``[B, n_fft//2+1, T, 2]``.
        n_fft: FFT size.
        hop_size: Hop length.
        win_size: Window length.
        center: Must be ``True`` (``False`` is not implemented).
        normalized: Whether the forward STFT was normalized.

    Returns:
        Reconstructed waveform ``[B, T_wav]``.
    """
    if not center:
        raise NotImplementedError("center=False is not implemented.",
            "Please use center=True to both stft & istft")
    
    global hann_window
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=spec.dtype, device=spec.device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav = torch.istft(spec, n_fft, hop_length=hop_size, win_length=win_size,
            center=center, normalized=normalized, window=hann_window[wnsize_dtype_device],
            onesided=True, return_complex=False)

    return wav


def spec_to_mel(spec: Tensor, n_fft: int, num_mels: int, sampling_rate: int,
                fmin: float = 0.0, fmax: Optional[float] = None, clip_val: float = 1e-5,
                log: bool = True, norm: str = 'slaney') -> Tensor:
    """Project a linear-frequency spectrogram to a mel-scale spectrogram.

    Args:
        spec: Magnitude spectrogram ``[B, n_fft//2+1, T]``.
        n_fft: FFT size used to produce *spec*.
        num_mels: Number of mel bands.
        sampling_rate: Audio sample rate in Hz.
        fmin: Lowest mel filter frequency in Hz.
        fmax: Highest mel filter frequency (``None`` = Nyquist).
        clip_val: Floor value before log (avoids log(0)).
        log: If ``True``, return log-mel.
        norm: Mel filterbank normalization (passed to librosa).

    Returns:
        Mel spectrogram ``[B, num_mels, T]``.
    """
    global mel_basis
    norm_dtype_device = str(norm) + '_' + str(spec.dtype) + '_' + str(spec.device)
    nmel_nfft_fmax_norm_dtype_device = str(num_mels) + '_' + str(n_fft) + '_' + str(fmax) + '_' + norm_dtype_device
    if nmel_nfft_fmax_norm_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax, norm=norm)
        mel_basis[nmel_nfft_fmax_norm_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[nmel_nfft_fmax_norm_dtype_device], spec)
    if log:
        spec = torch.log(torch.clamp(spec, min=clip_val))
    return spec


def mel_spectrogram(y: Tensor, n_fft: int, num_mels: int, sampling_rate: int, hop_size: int,
                    win_size: int, fmin: float = 0.0, fmax: Optional[float] = None,
                    center: bool = False, clip_val: float = 1e-5, log: bool = True,
                    norm: str = 'slaney') -> Tensor:
    """Compute mel spectrogram from raw waveform.

    Args:
        y: Input waveform ``[B, T]``.
        n_fft: FFT size.
        num_mels: Number of mel bands.
        sampling_rate: Sample rate in Hz.
        hop_size: Hop length.
        win_size: Window length.
        fmin: Lowest mel frequency in Hz.
        fmax: Highest mel frequency (``None`` = Nyquist).
        center: Whether to center-pad in STFT.
        clip_val: Floor value before log.
        log: If ``True``, return log-mel.
        norm: Mel filterbank normalization.

    Returns:
        Mel spectrogram ``[B, num_mels, mel_len]``.
    """
    spec = stft(y, n_fft, hop_size, win_size, center)
    mel = spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax, clip_val, log, norm)

    return mel