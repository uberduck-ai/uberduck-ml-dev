__all__ = [
    "pad_sequences",
    "prepare_input_sequence",
    "oversample",
    "TextMelCollate",
    "TextAudioSpeakerLoader",
    "TextAudioSpeakerCollate",
    "DistributedBucketSampler",
]


from typing import List

import numpy as np
import torch

from .text.symbols import (
    NVIDIA_TACO2_SYMBOLS,
)
from .text.util import text_to_sequence

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


# def map_text_encodings(encoding_1, encoding_2):
#     """Create a matrix associating graphemes/phonemes in encoding_1 with graphemes/phonemes in encoding_2."""
#     if set([encoding_1, encoding_2]) == set([TALKNET_SYMBOLS, NVIDIA_TACO2_SYMBOLS])
#         encoding_map = torch.zeros(len(TALKNET_SYMBOLS), len(NVIDIA_TACO2_SYMBOLS))
#         if encoding_2 == NVIDIA_TACO2_SYMBOLS:
#             return encoding_map.t()
#         else:
#             return encoding_map


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
                # NOTE (Sam): this adds a period to the end of every text.
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
