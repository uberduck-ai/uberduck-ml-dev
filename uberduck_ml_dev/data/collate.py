import torch
import numpy as np
from ..data.batch import Batch


class Collate:
    def __init__(
        self,
        n_frames_per_step: int = 1,
        cudnn_enabled: bool = False,
        # NOTE (Sam): these used to be computed in every batch.
        return_f0s: bool = False,
        return_mels: bool = True,
        return_text_sequences: bool = True,
        return_speaker_ids: bool = True,
        return_gsts: bool = True,
        return_audio_encodings: bool = False,
    ):
        self.n_frames_per_step = n_frames_per_step
        self.return_f0s = return_f0s
        self.return_mels = return_mels
        self.return_text_sequences = return_text_sequences
        self.return_speaker_ids = return_speaker_ids
        self.return_gsts = return_gsts
        self.return_audio_encodings = return_audio_encodings
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

    # TODO (Sam): don't return None-valued keys at all.
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        """

        input_lengths = torch.LongTensor([len(x["text_sequence"]) for x in batch])
        max_input_len = max(input_lengths)
        n_mel_channels = batch[0]["mel"].size(0)
        max_target_len = max([x["mel"].size(1) for x in batch])

        if self.return_text_sequences:
            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.zero_()
        else:
            text_padded = None

        if self.return_mels:
            mel_padded = torch.FloatTensor(len(batch), n_mel_channels, max_target_len)
            mel_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            gate_padded.zero_()
        else:
            mel_padded = None
            output_lengths = None
            gate_padded = None

        if self.return_speaker_ids:
            speaker_ids = torch.LongTensor(len(batch))
        else:
            speaker_ids = None
        if self.return_f0s:
            f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
            f0_padded.zero_()
        else:
            f0_padded = None
        if self.return_gsts:
            embedded_gsts = torch.FloatTensor(
                np.array([sample["embedded_gst"] for sample in batch])
            )
        else:
            embedded_gsts = None

        if self.return_audio_encodings:
            audio_encodings = torch.FloatTensor(
                torch.cat([sample["audio_encoding"] for sample in batch])
            )
        else:
            audio_encodings = None

        for i, sample in enumerate(batch):
            if self.return_mels:
                mel = sample["mel"]
                mel_padded[i, :, : mel.size(1)] = mel
                gate_padded[i, mel.size(1) - 1 :] = 1
                output_lengths[i] = mel.size(1)
            if self.return_speaker_ids:
                speaker_ids[i] = sample["speaker_id"]
            if self.return_f0s:
                f0_padded[i, :, : mel.size(1)] = sample["f0"]
            if self.return_text_sequences:
                text = sample["text_sequence"]
                text_padded[i, : text.size(0)] = text
            else:
                text_padded = None

        output = Batch(
            text_int_padded=text_padded,
            input_lengths=input_lengths,
            mel_padded=mel_padded,
            gate_target=gate_padded,
            output_lengths=output_lengths,
            speaker_ids=speaker_ids,
            audio_encodings=audio_encodings,
            gst=embedded_gsts,
            f0=f0_padded,
        )
        if self.cudnn_enabled:
            output = output.to_gpu()
        return output
