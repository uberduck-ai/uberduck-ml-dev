import torch
import numpy as np
from ..data.batch import Batch


class Collate:
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
        # NOTE (Sam): no longer requiring enforce_sorted but check dependent methods like SpeakerEncoder
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x["text_sequence"]) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]

        # NOTE (Sam): this reordering I believe is for compatibility with an earlier version of torch and should be removed.
        # for i in range(len(ids_sorted_decreasing)):
        #     text = batch[ids_sorted_decreasing[i]]["text_sequence"]
        #     text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        # num_mels = batch[0]["mel"].size(0 ) #what is this?
        max_target_len = max([x["mel"].size(1) for x in batch])

        if self.include_text:
            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.zero_()

        if self.include_mels:
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
            mel_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            gate_padded.zero_()

        if self.include_speaker_ids:
            speaker_ids = torch.LongTensor(len(batch))

        if self.include_f0:
            f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
            f0_padded.zero_()

        if self.include_embedded_gsts:
            embedded_gsts = torch.FloatTensor(
                np.array([sample["embedded_gst"] for sample in batch])
            )
        else:
            embedded_gsts = None

        # if "audio_encoding" in batch[0]:
        if self.include_audio_encodings:
            audio_encodings = torch.FloatTensor(
                torch.cat([sample["audio_encoding"] for sample in batch])
            )
        else:
            audio_encodings = None

        for i, sample in enumerate(batch):
            if self.include_mels:
                mel = sample["mel"]
                mel_padded[i, :, : mel.size(1)] = mel
                gate_padded[i, mel.size(1) - 1 :] = 1
                output_lengths[i] = mel.size(1)
                speaker_ids[i] = sample["speaker_id"]
            # if self.include_f0:

            # if self.include_embedded_gst:

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
