# TODO (Sam): use get to replace get_mels
from uberduck_ml_dev.data.get import get_mels
from uberduck_ml_dev.data.data import RADTTS_DEFAULTS as data_config
import os
import torch


class TestGetMels:
    def test_compute_mels_radtts(
        self,
        resampled_normalized_path_list,
        spectrogram_path_list,
        target_spectrogram_path_list,
    ):
        get(resampled_normalized_path_list, spectrogram_path_list)
        for sp, tsp in zip(
            resampled_normalized_path_list,
            spectrogram_path_list,
            target_spectrogram_path_list,
        ):
            assert os.path.exists(sp)
            assert torch.load(sp) == torch.load(tsp)

    def test_compute_mels_diffsinger(self, resampled_normalized_path_list,
        spectrogram_path_list,
        target_spectrogram_path_list):
