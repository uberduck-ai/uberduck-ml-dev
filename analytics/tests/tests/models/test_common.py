from uberduck_ml_dev.models.common import MelSTFT
import torch


class TestCommon:
    def test_mel_stft(self):
        mel_stft = MelSTFT()
        mel = mel_stft.mel_spectrogram(torch.clip(torch.randn(1, 1000), -1, 1))
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
