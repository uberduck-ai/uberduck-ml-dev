__all__ = [
    "pad_sequences",
    "prepare_input_sequence",
    "oversample",
    "TextMelDataset",
    "TextMelCollate",
    "TextAudioSpeakerLoader",
    "TextAudioSpeakerCollate",
    "DistributedBucketSampler",
]

import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
from scipy.io.wavfile import read
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .models.common import STFT, MelSTFT
from .text.symbols import (
    DEFAULT_SYMBOLS,
    IPA_SYMBOLS,
    NVIDIA_TACO2_SYMBOLS,
    GRAD_TTS_SYMBOLS,
)
from .text.util import cleaned_text_to_sequence, text_to_sequence
from .utils.audio import compute_yin, load_wav_to_torch
from .utils.utils import (
    load_filepaths_and_text,
    intersperse,
)
from .data.batch import Batch


def pad_sequences(batch):
    input_lengths = torch.LongTensor([len(x) for x in batch])
    max_input_len = input_lengths.max()

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(batch)):
        text = batch[i]
        text_padded[i, : text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(
    texts,
    cpu_run=False,
    arpabet=False,
    symbol_set=NVIDIA_TACO2_SYMBOLS,
    text_cleaner=["english_cleaners"],
):
    p_arpabet = float(arpabet)
    seqs = []
    for text in texts:
        seqs.append(
            torch.IntTensor(
                text_to_sequence(
                    text,
                    text_cleaner,
                    p_arpabet=p_arpabet,
                    symbol_set=symbol_set,
                )[:]
            )
        )
    text_padded, input_lengths = pad_sequences(seqs)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths


def oversample(filepaths_text_sid, sid_to_weight):
    assert all([isinstance(sid, str) for sid in sid_to_weight.keys()])
    output = []
    for fts in filepaths_text_sid:
        sid = fts[2]
        for _ in range(sid_to_weight.get(sid, 1)):
            output.append(fts)
    return output


def _orig_to_dense_speaker_id(speaker_ids):
    speaker_ids = np.asarray(list(set(speaker_ids)), dtype=str)
    id_order = np.argsort(np.asarray(speaker_ids, dtype=int))
    output = {
        orig: idx for orig, idx in zip(speaker_ids[id_order], range(len(speaker_ids)))
    }
    return output


class TextMelDataset(Dataset):
    def __init__(
        self,
        audiopaths_and_text: str,
        text_cleaners: List[str],
        p_arpabet: float,
        n_mel_channels: int,
        sampling_rate: int,
        mel_fmin: float,
        mel_fmax: float,
        filter_length: int,
        hop_length: int,
        win_length: int,
        symbol_set: str,
        padding: int = None,
        max_wav_value: float = 32768.0,
        include_f0: bool = False,
        pos_weight: float = 10,
        f0_min: int = 80,
        f0_max: int = 880,
        harmonic_thresh=0.25,
        debug: bool = False,
        debug_dataset_size: int = None,
        oversample_weights=None,
        intersperse_text: bool = False,
        intersperse_token: int = 0,
        compute_gst=None,
        audio_encoder_forward=None,
        speaker_embeddings=None,
    ):
        super().__init__()
        path = audiopaths_and_text
        oversample_weights = oversample_weights or {}
        self.audiopaths_and_text = oversample(
            load_filepaths_and_text(path), oversample_weights
        )
        self.text_cleaners = text_cleaners
        self.p_arpabet = p_arpabet

        self.stft = MelSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            padding=padding,
        )
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.include_f0 = include_f0
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.harmonic_threshold = harmonic_thresh
        # speaker id lookup table
        speaker_ids = [i[2] for i in self.audiopaths_and_text]
        self._speaker_id_map = _orig_to_dense_speaker_id(speaker_ids)
        self.debug = debug
        self.debug_dataset_size = debug_dataset_size
        self.symbol_set = symbol_set
        self.intersperse_text = intersperse_text
        self.intersperse_token = intersperse_token
        self.compute_gst = compute_gst
        self.audio_encoder_forward = audio_encoder_forward
        self.speaker_embeddings = speaker_embeddings

    def _get_f0(self, audio):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio,
            self.sampling_rate,
            self.filter_length,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            self.harmonic_threshold,
        )
        pad = int((self.filter_length / self.hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad
        f0 = np.array(f0, dtype=np.float32)
        return f0

    # TODO (Sam): rename this!
    def _get_gst(self, transcription):
        return self.compute_gst(transcription)

    def _get_audio_encoding(self, audio):
        return self.audio_encoder_forward(audio)

    def _get_data(self, audiopath_and_text):
        path, transcription, speaker_id = audiopath_and_text
        speaker_id = self._speaker_id_map[speaker_id]
        sampling_rate, wav_data = read(path)
        text_sequence = torch.LongTensor(
            text_to_sequence(
                transcription,
                self.text_cleaners,
                p_arpabet=self.p_arpabet,
                symbol_set=self.symbol_set,
            )
        )
        if self.intersperse_text:
            text_sequence = torch.LongTensor(
                intersperse(text_sequence.numpy(), self.intersperse_token)
            )  # add a blank token, whose id number is len(symbols)

        audio = torch.FloatTensor(wav_data)
        audio_norm = audio / (np.abs(audio).max() * 2)  # NOTE (Sam): just must be < 1.
        audio_norm = audio_norm.unsqueeze(0)

        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        data = {
            "text_sequence": text_sequence,
            "mel": melspec,
            "speaker_id": speaker_id,
            "f0": None,
        }

        if self.compute_gst:
            embedded_gst = self._get_gst([transcription])
            data["embedded_gst"] = embedded_gst

        if self.audio_encoder_forward is not None:
            # NOTE (Sam): hardcoded for debug
            audio_encoding = rearrange(self.speaker_embeddings, "o s -> 1 o s")
            data["audio_encoding"] = audio_encoding

        # NOTE (Sam): f0 not currently functional
        if self.include_f0:
            f0 = self._get_f0(audio.data.cpu().numpy())
            f0 = torch.from_numpy(f0)[None]
            f0 = f0[:, : melspec.size(1)]
            data["f0"] = f0

        return data

    def __getitem__(self, idx):
        """Return data for a single audio file + transcription."""
        try:
            data = self._get_data(self.audiopaths_and_text[idx])
        except Exception as e:
            print(f"Error while getting data: {self.audiopaths_and_text[idx]}")
            print(e)
            raise
        return data

    def __len__(self):
        if self.debug and self.debug_dataset_size:
            return min(self.debug_dataset_size, len(self.audiopaths_and_text))
        return len(self.audiopaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelCollate:
    def __init__(
        self,
        n_frames_per_step: int = 1,
        include_f0: bool = False,
        cudnn_enabled: bool = False,
    ):
        self.n_frames_per_step = n_frames_per_step
        self.include_f0 = include_f0
        self.cudnn_enabled = cudnn_enabled

    def set_frames_per_step(self, n_frames_per_step):
        """Set n_frames_step.

        This is used to train with gradual training, where we start with a large
        n_frames_per_step in order to learn attention quickly and decrease it
        over the course of training in order to increase accuracy. Gradual training
        reference:
        https://erogol.com/gradual-training-with-tacotron-for-faster-convergence/
        """
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x["text_sequence"]) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        # NOTE (Sam): this reordering I believe is for compatibility with an earlier version of torch and should be removed.
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]["text_sequence"]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0]["mel"].size(0)
        max_target_len = max([x["mel"].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        if self.include_f0:
            f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
            f0_padded.zero_()

        # NOTE (Sam): this reordering I believe is for compatibility with an earlier version of torch and should be removed.
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]["mel"]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]]["speaker_id"]

        # NOTE (Sam): does this make maximum sense?
        if "embedded_gst" in batch[0]:
            embedded_gsts = torch.FloatTensor(
                np.array([sample["embedded_gst"] for sample in batch])
            )
        else:
            embedded_gsts = None
        if "audio_encoding" in batch[0]:
            audio_encodings = torch.FloatTensor(
                torch.cat([sample["audio_encoding"] for sample in batch])
            )
        else:
            audio_encodings = None
        output = Batch(
            text_int_padded=text_padded,
            input_lengths=input_lengths,
            mel_padded=mel_padded,
            gate_target=gate_padded,
            output_lengths=output_lengths,
            speaker_ids=speaker_ids,
            audio_encodings=audio_encodings,
            gst=embedded_gsts,
        )
        if self.cudnn_enabled:
            output = output.to_gpu()
        return output
