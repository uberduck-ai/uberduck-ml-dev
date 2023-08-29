import librosa
import numpy as np
from scipy.io.wavfile import write
from ..models.tacotron2 import MAX_WAV_VALUE

load_resampled_normalized_audio = lambda source_path: librosa.load(
    source_path, sr=22050
)[0]
float_normalize = lambda x: np.asarray(
    (x / np.abs(x).max()) * (MAX_WAV_VALUE - 1) / MAX_WAV_VALUE
)
int_normalize = lambda x: np.asarray(
    (x / np.abs(x).max()) * (MAX_WAV_VALUE - 1), dtype=np.int16
)
save_22k_audio = lambda data, target_path: write(
    target_path, 22050, data
)  # must be in this order
