import torch
import os
from typing import List
from torch.utils.data import Dataset
from einops import rearrange
from scipy.io.wavfile import read
from scipy.ndimage import distance_transform_edt as distance_transform
import numpy as np
from librosa import pyin

from ..models.common import MelSTFT
from ..utils.utils import (
    load_filepaths_and_text,
    intersperse,
)
from .utils import oversample, _orig_to_dense_speaker_id
from ..text.util import text_to_sequence


class Data(Dataset):
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
        use_f0=False,
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
        self.use_f0 = use_f0
        
    # NOTE (Sam): this is the RADTTS version.
    # RADTTS is more recent than mellotron from the same author so let's assume this method is better.
    def _get_f0(self, audiopath, audio):
        filename = "_".join(audiopath.split("/")[-3:])
        f0_path = os.path.join(self.betabinom_cache_path, filename)
        f0_path += "_f0_sr{}_fl{}_hl{}_f0min{}_f0max{}_log{}.pt".format(
            self.sampling_rate,
            self.filter_length,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            self.use_log_f0,
        )

        dikt = None
        if os.path.exists(f0_path):
            try:
                dikt = torch.load(f0_path)
            except:
                print(f"f0 loading from {f0_path} is broken, recomputing.")

        if dikt is not None:
            f0 = dikt["f0"]
            p_voiced = dikt["p_voiced"]
            voiced_mask = dikt["voiced_mask"]
        else:
            f0, voiced_mask, p_voiced = self.get_f0_pvoiced(
                audio.cpu().numpy(),
                self.sampling_rate,
                self.filter_length,
                self.hop_length,
                self.f0_min,
                self.f0_max,
            )
            print("saving f0 to {}".format(f0_path))
            torch.save(
                {"f0": f0, "voiced_mask": voiced_mask, "p_voiced": p_voiced},
                f0_path,
            )
        if f0 is None:
            raise Exception("STOP, BROKEN F0 {}".format(audiopath))

        f0 = self.f0_normalize(f0)
        if self.distance_tx_unvoiced:
            mask = f0 <= 0.0
            distance_map = np.log(distance_transform(mask))
            distance_map[distance_map <= 0] = 0.0
            f0 = f0 - distance_map

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

        # TODO (Sam): treat these covariates more equivalently
        f0 = None
        if self.use_f0:
            f0 = self._get_f0(self, audio)

        data = {
            "text_sequence": text_sequence,
            "mel": melspec,
            "speaker_id": speaker_id,
            "f0": f0,
        }

        if self.compute_gst:
            embedded_gst = self._get_gst([transcription])
            data["embedded_gst"] = embedded_gst

        if self.audio_encoder_forward is not None:
            # NOTE (Sam): hardcoded for now.
            audio_encoding = rearrange(self.speaker_embeddings, "o s -> 1 o s")
            data["audio_encoding"] = audio_encoding

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

    def get_f0_pvoiced(
        self,
        audio,
        sampling_rate=22050,
        frame_length=1024,
        hop_length=256,
        f0_min=100,
        f0_max=300,
    ):

        audio_norm = audio / self.max_wav_value
        f0, voiced_mask, p_voiced = pyin(
            audio_norm,
            f0_min,
            f0_max,
            sampling_rate,
            frame_length=frame_length,
            win_length=frame_length // 2,
            hop_length=hop_length,
        )
        f0[~voiced_mask] = 0.0
        f0 = torch.FloatTensor(f0)
        p_voiced = torch.FloatTensor(p_voiced)
        voiced_mask = torch.FloatTensor(voiced_mask)
        return f0, voiced_mask, p_voiced
