from scipy.io.wavfile import read
from uberduck_ml_dev.models.common import MelSTFT
import torch


class TestHifiGan:
    def test_hifi_gan(self):
        # TODO (Sam): move to settings file.
        path = "analytics/tests/fixtures/wavs/stevejobs-1.wav"
        sr, data = read(path)

        assert sr == 22050
        assert len(data) == 144649

        data = torch.FloatTensor(data / 32768.0).unsqueeze(0)

        melstft = MelSTFT()
        mel = melstft.mel_spectrogram(data)

        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
        assert mel.shape[2] == 566
