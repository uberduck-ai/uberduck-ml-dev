import os
from scipy.io.wavfile import read, write
import librosa
import torch
import numpy as np

from uberduck_ml_dev.data.get import get
from uberduck_ml_dev.data.utils import mel_spectrogram_torch, find_rel_paths
from uberduck_ml_dev.data.data import HIFIGAN_DEFAULTS as DEFAULTS
from uberduck_ml_dev.data.data import MAX_WAV_VALUE


data_directory = ""  # path to the directory containing the data
ground_truth_rel_paths = find_rel_paths(directory=data_directory, filename="gt.wav")
ground_truth_abs_paths = [
    os.path.join(data_directory, ground_truth_rel_path)
    for ground_truth_rel_path in ground_truth_rel_paths
]


print("resampling and integer normalizing")

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
    True,
)

print("resampling and float normalizing")

resampled_normalized_abs_paths = [
    resampled_normalized_abs_path.replace("gt.wav", "audio_resampledT_normalized1T.wav")
    for resampled_normalized_abs_path in ground_truth_abs_paths
]

loading_function = lambda filename: librosa.load(filename, sr=22050)[0]
processing_function = lambda x: np.asarray(
    (x / np.abs(x).max()) * (1 - 1 / MAX_WAV_VALUE), dtype=np.float32
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
    True,
)


print("computing spectrograms from 1 normalized audio")

spectrogram_abs_paths = [
    ground_truth_abs_path.replace("gt.wav", "spectrogram.pt")
    for ground_truth_abs_path in ground_truth_abs_paths
]


processing_function = lambda x: mel_spectrogram_torch(
    x,
    DEFAULTS["n_fft"],
    DEFAULTS["num_mels"],
    DEFAULTS["sampling_rate"],
    DEFAULTS["hop_size"],
    DEFAULTS["win_size"],
    DEFAULTS["fmin"],
    DEFAULTS["fmax"],
    True,
)
loading_function = lambda source_path: torch.Tensor(
    read(source_path)[1] / MAX_WAV_VALUE
).unsqueeze(0)
saving_function = lambda data, target_path: torch.save(data, target_path)

get(
    processing_function,
    saving_function,
    loading_function,
    resampled_normalized_abs_paths,
    spectrogram_abs_paths,
    True,
)
