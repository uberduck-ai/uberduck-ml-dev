import torch
from librosa.filters import mel as librosa_mel_fn

from .utils import spectral_normalize_torch


# TODO (Sam): combine with identically-named function is models.common
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram.

    Args:
        y             :: (B, T) - Audio waveforms
        n_fft
        sampling_rate
        hop_size
        win_size
        center
    Returns:
        :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """
    # Validation
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    # Window - Cache if needed
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Padding
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        # NOTE (Sam): combinining n_fft (filter_length) with hop_size reeks of either a bug or sophisticated asympotitc analysis.
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Complex Spectrogram :: (B, T) -> (B, Freq, Frame, RealComplex=2)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # MelBasis - Cache if needed
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq=num_mels, Frame)
    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """Convert waveform into Mel-frequency Log-amplitude spectrogram.

    Args:
        y       :: (B, T)           - Waveforms
    Returns:
        melspec :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
    """
    # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

    return melspec
