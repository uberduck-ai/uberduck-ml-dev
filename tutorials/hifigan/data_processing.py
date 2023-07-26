import os
from scipy.io.wavfile import read, write
import librosa
import torch
import numpy as np

from uberduck_ml_dev.data.get import get
from uberduck_ml_dev.data.utils import mel_spectrogram_torch, find_rel_paths
from uberduck_ml_dev.data.data import HIFIGAN_DEFAULTS as DEFAULTS
from uberduck_ml_dev.data.data import MAX_WAV_VALUE


data_directory = "/usr/src/app/uberduck_ml_exp/data/lj_redo/"
ground_truth_rel_paths = find_rel_paths(directory=data_directory, filename="gt.wav")
ground_truth_abs_paths = [
    os.path.join(data_directory, ground_truth_rel_path)
    for ground_truth_rel_path in ground_truth_rel_paths
]


print("resampling and integer/32768 normalizing spectrograms")

resampled_normalized_abs_paths = [
    resampled_normalized_abs_path.replace(
        "gt.wav", "audio_resampledT_normalized32768T.wav"
    )
    for resampled_normalized_abs_path in ground_truth_abs_paths
]

loading_function = lambda filename: librosa.load(filename, sr=22050)[0]
processing_function = lambda x: np.asarray(
    (x / np.abs(x).max()) * (MAX_WAV_VALUE - 1), dtype=np.int16
)
saving_function = lambda data, filename: write(
    filename, 22050, data
)  # must be in this order

get(
    processing_function,
    saving_function,
    loading_function,
    ground_truth_abs_paths,
    resampled_normalized_abs_paths,
    False,
)


print("compute spectrograms")

spectrogram_abs_paths = [
    ground_truth_abs_path.replace("gt.wav", "spectrogram.pt")
    for ground_truth_abs_path in ground_truth_abs_paths
]
processing_function = lambda x: mel_spectrogram_torch(
    x,
    DEFAULTS["n_fft"],
    DEFAULTS["sampling_rate"],
    DEFAULTS["hop_size"],
    DEFAULTS["win_size"],
    True,
)
loading_function = lambda source_path: read(source_path)[1]
saving_function = lambda data, target_path: torch.save(target_path, data)

get(
    processing_function,
    saving_function,
    loading_function,
    resampled_normalized_abs_paths,
    spectrogram_abs_paths,
    False,
)
