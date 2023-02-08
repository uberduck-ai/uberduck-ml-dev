import torch
import os
from typing import List, Optional, Dict
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
from ..models.common import FILTER_LENGTH, HOP_LENGTH, WIN_LENGTH
from ..models.tacotron2 import N_MEL_CHANNELS
from ..text.symbols import NVIDIA_TACO2_SYMBOLS

# NOTE (Sam): generic dataset class for all purposes avoids writing redundant methods (e.g. get pitch when text isn't available)
# However, factor out components of this dataloader (e.g. get mels, get pitch) and merging classes as needed would be preferable.
# e.g. https://www.daniweb.com/programming/software-development/code/283396/merging-class-instances although I'm not quite sure.
# NOTE (Sam): "load" means load from file, "compute" means compute in dataloader.
class Data(Dataset):
    def __init__(
        self,
        audiopaths_and_text: Optional[
            str
        ] = None,  # TODO (Sam): consider removing triplicate audiopaths_and_text argument
        oversample_weights=None,
        # Text parameters
        return_texts: bool = False,  # NOTE (Sam): maybe include include_texts parameter if text is ever inferred.
        texts: Optional[List[str]] = None,
        intersperse_text: Optional[bool] = False,
        intersperse_token: Optional[int] = 0,
        text_cleaners: Optional[List[str]] = "english_cleaners",
        p_arpabet: Optional[float] = 1.0,
        # Audio parameters
        return_mels=True,
        audiopaths: Optional[List[str]] = None,
        load_audios: bool = False,
        n_mel_channels: Optional[int] = N_MEL_CHANNELS,
        sampling_rate: Optional[int] = 22050,
        mel_fmin: Optional[float] = 0.0,
        mel_fmax: Optional[float] = 8000,
        filter_length: Optional[int] = FILTER_LENGTH,
        hop_length: Optional[int] = HOP_LENGTH,
        win_length: Optional[int] = WIN_LENGTH,
        symbol_set: Optional[str] = NVIDIA_TACO2_SYMBOLS,
        padding: Optional[int] = None,
        max_wav_value: Optional[float] = 32768.0,
        # Pitch parameters
        # TODO (Sam): consider use_f0 = load_f0 or compute_f0
        return_f0s: bool = False,
        load_f0s: bool = False,
        f0_min: Optional[int] = 80,
        f0_max: Optional[int] = 880,
        # Torchmoji parameters
        return_gsts: bool = False,
        load_gsts=False,  # TODO (Sam): check this against existing crust models
        get_gst=None,  # NOTE (Sam): this is a functional argument.
        # Speaker embedding parameters
        return_speaker_ids: bool = True,
        load_speaker_ids: bool = True,
        speaker_ids: Optional[List[str]] = None,
        # TODO (Sam): add include/compute syntax to these embeddings.  They are an alternative to load_speaker_ids
        audio_encoder_forward=None,
        speaker_embeddings=None,
        # Control parameters
        debug: bool = False,
        debug_dataset_size: int = None,  # NOTE (Sam): is this optional?
        # oversample_weights: Optional[Dict] = None,  # TODO (Sam): type this.
    ):
        super().__init__()
        self.debug = debug
        self.debug_dataset_size = debug_dataset_size

        # TODO (Sam): refactor support for oversampling to make generic across data types.
        if audiopaths_and_text:
            oversample_weights = {}
            self.audiopaths_and_text = oversample(
                load_filepaths_and_text(audiopaths_and_text), oversample_weights
            )

        self.return_texts = return_texts
        self.return_mels = return_mels
        self.return_f0s = return_f0s
        self.load_f0s = load_f0s
        self.return_gsts = return_gsts
        self.load_gsts = load_gsts
        self.return_speaker_ids = return_speaker_ids
        self.load_speaker_ids = load_speaker_ids

        if self.return_mels:
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

            if self.audiopaths_and_text:
                self.audiopaths = [i[0] for i in self.audiopaths_and_text]
            else:
                self.audiopaths = audiopaths

        if self.return_texts:
            self.text_cleaners = text_cleaners
            self.p_arpabet = p_arpabet
            self.symbol_set = symbol_set
            self.intersperse_text = intersperse_text
            self.intersperse_token = intersperse_token
            if self.audiopaths_and_text:
                self.texts = [i[1] for i in self.audiopaths_and_text]
            else:
                self.texts = texts

        if self.return_f0s:
            # NOTE (Sam): its unclear if these are necessary for load_f0 only
            self.f0_min = f0_min
            self.f0_max = f0_max

        if self.return_gsts and not self.load_gsts:
            self.get_gst = get_gst

        # NOTE (Sam): right now only old audiopaths_and_text based loading is supported.
        # TODO (Sam): think more carefully about how the audio_encoder interface should work.
        if self.return_speaker_ids:
            if self.load_speaker_ids:
                if self.audiopaths_and_text:
                    speaker_ids = [i[2] for i in self.audiopaths_and_text]
                    self._speaker_id_map = _orig_to_dense_speaker_id(speaker_ids)
            # else could be speaker classification positions for example.

        # NOTE (Sam): this is hacky and not consistent with other approaches.
        self.audio_encoder_forward = audio_encoder_forward
        self.speaker_embeddings = speaker_embeddings

    # NOTE (Sam): this is the RADTTS version - more recent than mellotron from the same author.
    # NOTE (Sam): in contrast to get_gst, the computation here is kept in this file rather than a functional argument.
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
    def _get_gst(self, text):
        return self.get_gst(text)

    def _get_audio_encoding(self, audio):
        return self.audio_encoder_forward(audio)

    def _get_data(
        self,
        audiopath_and_text: Optional[List[str]] = None,
        audiopath: Optional[str] = None,
        text: Optional[str] = None,
        speaker_id: Optional[int] = None,
    ):
        data = {}
        if audiopath_and_text is not None:
            audiopath, text, speaker_id = audiopath_and_text
        speaker_id = self._speaker_id_map[speaker_id]

        if self.return_texts:
            text_sequence = torch.LongTensor(
                text_to_sequence(
                    text,
                    self.text_cleaners,
                    p_arpabet=self.p_arpabet,
                    symbol_set=self.symbol_set,
                )
            )
            if self.intersperse_text:
                text_sequence = torch.LongTensor(
                    intersperse(text_sequence.numpy(), self.intersperse_token)
                )  # add a blank token, whose id number is len(symbols)
            data["text_sequence"] = text_sequence

        if self.return_mels:
            sampling_rate, wav_data = read(audiopath)
            # NOTE (Sam): is this the right normalization?  Should it be done here or in preprocessing.
            audio = torch.FloatTensor(wav_data)
            audio_norm = audio / (
                np.abs(audio).max() * 2
            )  # NOTE (Sam): just must be < 1.
            audio_norm = audio_norm.unsqueeze(0)

            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            data["mel"] = melspec

        # TODO (Sam): treat these covariates more equivalently
        f0 = None
        if self.return_f0s:
            if not self.load_f0s:
                f0 = self._get_f0(self, audio)
                data["f0"] = f0

        if self.return_speaker_ids:
            data["speaker_id"] = speaker_id

        if self.return_gsts:
            if not self.load_gsts:
                embedded_gst = self._get_gst([text])
                data["embedded_gst"] = embedded_gst

        if self.audio_encoder_forward is not None:
            # NOTE (Sam): hardcoded for now.
            audio_encoding = rearrange(self.speaker_embeddings, "o s -> 1 o s")
            data["audio_encoding"] = audio_encoding

        return data

    def __getitem__(self, idx):
        """Return data for a single audio file + transcription."""
        try:
            if self.audiopaths_and_text:
                data = self._get_data(self.audiopaths_and_text[idx])
            else:
                data = self._get_data(
                    audiopath=self.audiopaths[idx],
                    text=self.texts[idx],
                    speaker_id=self.speaker_ids[idx],
                )
        except Exception as e:
            print(f"Error while getting data: index = {idx}")
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
