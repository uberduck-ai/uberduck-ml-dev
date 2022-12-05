"""
Removes bias from vocoders (typically heard as metalic noise in the audio)

Usage:
from denoiser import Denoiser
denoiser = Denoiser(VOCODERGENERATOR, mode="normal") # Experiment with modes "normal" and "zeros"

# Inference Vocoder
audio = VOCODERGENERATOR.vocoder.forward(output[1][:1])

# Denoise
audio_denoised = denoiser(audio.view(1, -1), strength=15)[:, 0] # Change strength if needed

audio_denoised = audio_denoised.cpu().detach().numpy().reshape(-1)
normalize = (32768.0 / np.max(np.abs(audio_denoised))) ** 0.9
audio_denoised = audio_denoised * normalize
"""

import sys
import torch
from ..models.common import STFT
from ..vocoders.istftnet import iSTFTNetGenerator


class Denoiser(torch.nn.Module):
    """WaveGlow denoiser, adapted for HiFi-GAN"""

    def __init__(
        self, hifigan, filter_length=1024, n_overlap=4, win_length=1024, mode="zeros"
    ):
        super(Denoiser, self).__init__()
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
            device=torch.device("cpu"),
        )

        if mode == "zeros":
            mel_input = torch.zeros((1, 80, 88))
        elif mode == "normal":
            mel_input = torch.randn((1, 80, 88))
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            if isinstance(hifigan, iSTFTNetGenerator):
                bias_audio = (
                    hifigan(mel_input.to(hifigan.device))
                    .view(1, -1)
                    .float()
                )
            else:
                bias_audio = (
                    hifigan.vocoder.forward(mel_input.to(hifigan.device))
                    .view(1, -1)
                    .float()
                )
            bias_spec, _ = self.stft.transform(bias_audio.cpu())

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=10):
        """
        Strength is the amount of bias you want to be removed from the final audio.
        Note: A higher strength may remove too much information in the original audio.

        :param audio: Audio data
        :param strength: Amount of bias removal. Recommended range 10 - 50
        :return: Denoised audio
        :rtype: tensor
        """

        audio_spec, audio_angles = self.stft.transform(audio.cpu())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised.cpu(), audio_angles)
        return audio_denoised
