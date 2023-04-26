import torch
import numpy as np
from ..data.batch import Batch


class Collate:
    def __init__(
        self,
        n_frames_per_step: int = 1,
        cudnn_enabled: bool = False,
    ):
        self.n_frames_per_step = n_frames_per_step
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
        """Collate training batch from normalized text and mel-spectrogram.
        PARAMS
        ------
        """

        return_f0s = "f0" in batch[0]
        return_mels = "mel" in batch[0]
        return_text_sequences = "text_sequence" in batch[0]
        return_speaker_ids = "speaker_id" in batch[0]
        return_gsts = "embedded_gst" in batch[0]
        return_audio_encodings = "audio_encoding" in batch[0]

        if return_text_sequences:
            input_lengths = torch.LongTensor([len(x["text_sequence"]) for x in batch])
            max_input_len = max(input_lengths)
            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.zero_()
        else:
            text_padded = None
            input_lengths = None
        if return_mels:
            n_mel_channels = batch[0]["mel"].size(0)
            max_target_len = max([x["mel"].size(1) for x in batch])
            mel_padded = torch.FloatTensor(len(batch), n_mel_channels, max_target_len)
            mel_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            gate_padded.zero_()
        else:
            mel_padded = None
            output_lengths = None
            gate_padded = None

        if return_speaker_ids:
            speaker_ids = torch.LongTensor(len(batch))
        else:
            speaker_ids = None
        if return_f0s:
            max_target_len = max([x["f0"].size(0) for x in batch])
            f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
            f0_padded.zero_()
        else:
            f0_padded = None
        if return_gsts:
            embedded_gsts = torch.FloatTensor(
                np.array([sample["embedded_gst"] for sample in batch])
            )
        else:
            embedded_gsts = None

        if return_audio_encodings:
            audio_encodings = torch.FloatTensor(
                torch.cat([sample["audio_encoding"] for sample in batch])
            )
        else:
            audio_encodings = None

        for i, sample in enumerate(batch):
            if return_mels:
                mel = sample["mel"]
                mel_padded[i, :, : mel.size(1)] = mel
                gate_padded[i, mel.size(1) - 1 :] = 1
                output_lengths[i] = mel.size(1)
            if return_speaker_ids:
                speaker_ids[i] = sample["speaker_id"]
            if return_f0s:
                f0 = sample["f0"]
                f0_padded[i, :, : f0.size(0)] = f0
            if return_text_sequences:
                text = sample["text_sequence"]
                text_padded[i, : text.size(0)] = text

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


# TODO (Sam): combine Collate with DataCollateRADTTS
class DataCollateRADTTS:
    """Zero-pads model inputs and targets given number of steps"""

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate from normalized data"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x["text_encoded"]) for x in batch]),
            dim=0,
            descending=True,
        )

        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]["text_encoded"]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mel_channels = batch[0]["mel"].size(0)
        max_target_len = max([x["mel"].size(1) for x in batch])

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mel_channels, max_target_len)
        mel_padded.zero_()
        f0_padded = None
        p_voiced_padded = None
        voiced_mask_padded = None
        energy_avg_padded = None
        if batch[0]["f0"] is not None:
            f0_padded = torch.FloatTensor(len(batch), max_target_len)
            f0_padded.zero_()

        if batch[0]["p_voiced"] is not None:
            p_voiced_padded = torch.FloatTensor(len(batch), max_target_len)
            p_voiced_padded.zero_()

        if batch[0]["voiced_mask"] is not None:
            voiced_mask_padded = torch.FloatTensor(len(batch), max_target_len)
            voiced_mask_padded.zero_()

        if batch[0]["energy_avg"] is not None:
            energy_avg_padded = torch.FloatTensor(len(batch), max_target_len)
            energy_avg_padded.zero_()

        if batch[0]["audio_embedding"] is not None:
            audio_embedding_padded = torch.FloatTensor(
                len(batch), 512
            )  # emb size - TODO (Sam): try to reduce this via PCA
            # audio_embedding_padded = torch.FloatTensor(len(batch), 16) # PCA version
            audio_embedding_padded.zero_()

        attn_prior_padded = torch.FloatTensor(len(batch), max_target_len, max_input_len)
        attn_prior_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        audiopaths = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]["mel"]
            mel_padded[i, :, : mel.size(1)] = mel

            if batch[ids_sorted_decreasing[i]]["audio_embedding"] is not None:
                audio_embedding_padded[i, :] = batch[ids_sorted_decreasing[i]][
                    "audio_embedding"
                ]

            if batch[ids_sorted_decreasing[i]]["f0"] is not None:
                f0 = batch[ids_sorted_decreasing[i]]["f0"]
                f0_padded[i, : len(f0)] = f0

            if batch[ids_sorted_decreasing[i]]["voiced_mask"] is not None:
                voiced_mask = batch[ids_sorted_decreasing[i]]["voiced_mask"]
                voiced_mask_padded[i, : len(f0)] = voiced_mask

            if batch[ids_sorted_decreasing[i]]["p_voiced"] is not None:
                p_voiced = batch[ids_sorted_decreasing[i]]["p_voiced"]
                p_voiced_padded[i, : len(f0)] = p_voiced

            if batch[ids_sorted_decreasing[i]]["energy_avg"] is not None:
                energy_avg = batch[ids_sorted_decreasing[i]]["energy_avg"]
                energy_avg_padded[i, : len(energy_avg)] = energy_avg

            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]]["speaker_id"]
            audiopath = batch[ids_sorted_decreasing[i]]["audiopath"]
            audiopaths.append(audiopath)
            cur_attn_prior = batch[ids_sorted_decreasing[i]]["attn_prior"]
            if cur_attn_prior is None:
                attn_prior_padded = None
            else:
                attn_prior_padded[
                    i, : cur_attn_prior.size(0), : cur_attn_prior.size(1)
                ] = cur_attn_prior

        return {
            "mel": mel_padded,
            "speaker_ids": speaker_ids,
            "text": text_padded,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "audiopaths": audiopaths,
            "attn_prior": attn_prior_padded,
            "f0": f0_padded,
            "p_voiced": p_voiced_padded,
            "voiced_mask": voiced_mask_padded,
            "energy_avg": energy_avg_padded,
            "audio_embedding": audio_embedding_padded,
        }


# NOTE (Sam): this is a dummy collate function for torch-based parallel processing.
# This appears faster than ray data or multiprocessing parallel processing.
# I'm not 100% sure.
class CollateBlank:
    def __init__(
        self,
    ):
        2 + 2

    def __call__(self, batch):
        return None
