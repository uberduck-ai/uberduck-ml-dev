# Adapted from https://github.com/coqui-ai/TTS/blob/dev/TTS/encoder/models/base_encoder.py

import numpy as np
import torch
import torchaudio

from torch import nn


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "filter",
            torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class BaseEncoder(nn.Module):
    """Base `encoder` class. Every new `encoder` model must inherit this.

    It defines common `encoder` specific functions.
    """

    # pylint: disable=W0102
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            # TorchSTFT(
            #     n_fft=audio_config["fft_size"],
            #     hop_length=audio_config["hop_length"],
            #     win_length=audio_config["win_length"],
            #     sample_rate=audio_config["sample_rate"],
            #     window="hamming_window",
            #     mel_fmin=0.0,
            #     mel_fmax=None,
            #     use_htk=True,
            #     do_amp_to_db=False,
            #     n_mels=audio_config["num_mels"],
            #     power=2.0,
            #     use_mel=True,
            #     mel_norm=None,
            # )
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    @torch.no_grad()
    def inference(self, x, l2_norm=True):
        return self.forward(x, l2_norm)

    @torch.no_grad()
    def compute_embedding(
        self, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True
    ):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        # map to the waveform size
        if self.use_torch_spec:
            num_frames = num_frames * self.audio_config["hop_length"]

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings
