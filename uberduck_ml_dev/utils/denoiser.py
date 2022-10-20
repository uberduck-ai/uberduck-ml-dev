import torch
from ..models.common import STFT

"""
Removes bias from HiFi-Gan and Avocodo (typically heard as noise in the audio)

Usage:
from denoiser import Denoiser
denoiser = Denoiser(HIFIGANGENERATOR, mode="normal", device=device)

# Inference Vocoder
audio = hifigan.vocoder.forward(output[1][:1])

audio = audio.squeeze()
audio = audio * 32768.0

# Denoise
audio_denoised = denoiser(audio.view(1, -1), strength=15)[:, 0]

audio_denoised = audio_denoised.cpu().detach().numpy().reshape(-1)
normalize = (32768.0 / np.max(np.abs(audio_denoised))) ** 0.9
audio_denoised = audio_denoised * normalize
"""

class Denoiser(torch.nn.Module):
    """ WaveGlow denoiser, adapted for HiFi-GAN """

    def __init__(
        self, hifigan, filter_length=1024, n_overlap=4, win_length=1024, mode="zeros", device="cpu"
    ):
        super(Denoiser, self).__init__()
        self.device = device
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
            device=torch.device(self.device),
        )

        if mode == "zeros":
            mel_input = torch.zeros((1, 80, 88)).to(torch.device(self.device))
        elif mode == "normal":
            mel_input = torch.randn((1, 80, 88)).to(torch.device(self.device))
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = hifigan.vocoder.forward(mel_input).view(1, -1).float()
            bias_spec, _ = self.stft.transform(bias_audio, device=self.device)

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.to(torch.device(self.device)).float(), device=self.device)
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles, device=self.device)
        return audio_denoised
