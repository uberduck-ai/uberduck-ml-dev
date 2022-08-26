import json
import random

from einops import rearrange
import torch
import numpy as np

from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS
from uberduck_ml_dev.models.tacotron2 import Tacotron2
from uberduck_ml_dev.trainer.tacotron2 import Tacotron2Trainer
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
import torch
from collections import Counter


class TestTacotron2Model:
    def test_tacotron2_model(self):
        config = TACOTRON2_DEFAULTS.values()
        with open("tests/fixtures/ljtest/taco2_lj2lj.json") as f:
            config.update(json.load(f))
        hparams = HParams(**config)
        hparams.speaker_embedding_dim = 1
        model = Tacotron2(hparams)
        if torch.cuda.is_available() and hparams.cudnn_enabled:
            model.cuda()
        trainer = Tacotron2Trainer(hparams, rank=0, world_size=0)
        (
            train_set,
            val_set,
            train_loader,
            sampler,
            collate_fn,
        ) = trainer.initialize_loader()
        batch = next(enumerate(train_loader))[1]
        # NOTE (Sam): call subsets directly in function arguments
        model_input = batch.subset(
            [
                "text_int_padded",
                "input_lengths",
                "speaker_ids",
                "gst",
                "mel_padded",
                "output_lengths",
            ]
        )
        model_output = model(
            input_text=model_input["text_int_padded"],
            input_lengths=model_input["input_lengths"],
            speaker_ids=model_input["speaker_ids"],
            embedded_gst=model_input["gst"],
            targets=model_input["mel_padded"],
            output_lengths=model_input["output_lengths"],
        )

        # 'mel_outputs', 'mel_outputs_postnet', 'gate_predicted', 'output_lengths', 'alignments'
        assert len(model_output) == 5

    def test_stft_seed(self, sample_inference_spectrogram, lj_speech_tacotron2):

        torch.random.manual_seed(1234)
        np.random.seed(1234)
        nreps = 1
        text = "I, Sam, am a very good boy."
        lines = [t for t in text.split("\n") if t]
        sequences, input_lengths = prepare_input_sequence(
            lines,
            cpu_run=True,
            arpabet=True,
            symbol_set="nvidia_taco2",
            text_cleaner=["english_cleaners"],
        )
        sequences = sequences.repeat(nreps, 1)
        input_lengths = input_lengths.repeat(1, nreps).squeeze(0)

        mel_outputs_postnet = lj_speech_tacotron2.inference(sequences, input_lengths)[
            "mel_outputs_postnet"
        ]

        estimated_vector = rearrange(
            mel_outputs_postnet.detach().numpy(), "b m t -> (b m t)"
        )
        target_vector = rearrange(
            sample_inference_spectrogram.detach().numpy(), "m t -> (m t)"
        )
        rho = np.corrcoef(
            estimated_vector,
            target_vector,
        )[0, 1]
        assert rho > 0.99

        torch.random.manual_seed(1235)
        np.random.seed(1235)

        mel_outputs_postnet = lj_speech_tacotron2.inference(sequences, input_lengths)[
            "mel_outputs_postnet"
        ]

        tf_index = 111  # NOTE (Sam): look at the beginning of the clip since they are different lengths.
        estimated_vector = rearrange(
            mel_outputs_postnet.detach().numpy()[:, :, :tf_index], "b m t -> (b m t)"
        )
        target_vector = rearrange(
            sample_inference_spectrogram.detach().numpy()[:, :tf_index],
            "m t -> (m t)",
        )
        rho = np.corrcoef(
            estimated_vector,
            target_vector,
        )[0, 1]
        assert rho < 0.98

    def test_tf_inference(
        self,
        lj_speech_tacotron2,
        sample_inference_spectrogram,
        sample_inference_tf_spectrogram,
    ):

        torch.random.manual_seed(1234)
        np.random.seed(1234)
        tf_index = 111  # NOTE (Sam): this was determined by listening to the output
        nreps = 1
        text = "I, Sam, am a very bad boy."
        lines = [t for t in text.split("\n") if t]
        sequences, input_lengths = prepare_input_sequence(
            lines,
            cpu_run=True,
            arpabet=True,
            symbol_set="nvidia_taco2",
            text_cleaner=["english_cleaners"],
        )
        sequences = sequences.repeat(nreps, 1)
        input_lengths = input_lengths.repeat(1, nreps).squeeze(0)
        sample_inference_spectrogram = sample_inference_spectrogram[None, :]

        mel_outputs_postnet_tf = lj_speech_tacotron2.inference_partial_tf(
            input_text=sequences,
            input_lengths=input_lengths,
            speaker_ids=None,
            embedded_gst=None,
            tf_mel=sample_inference_spectrogram,
            tf_until_idx=tf_index,
        )["mel_outputs_postnet"]

        mel_outputs_postnet_original = lj_speech_tacotron2.inference(
            input_text=sequences, input_lengths=input_lengths, speaker_ids=None
        )["mel_outputs_postnet"]

        non_tf_mel_outputs_postnet_beginning = (
            mel_outputs_postnet_original.detach().numpy()[:, :, :tf_index]
        )

        tf_mel_outputs_postnet_beginning = mel_outputs_postnet_tf.detach().numpy()[
            :, :, :tf_index
        ]
        sample_inference_spectrogram_beginning = (
            sample_inference_spectrogram.detach().numpy()[:, :, :tf_index]
        )

        original_vector_beginning = rearrange(
            sample_inference_spectrogram_beginning, "b m t -> (b m t)"
        )

        tf_estimate_vector_beginning = rearrange(
            tf_mel_outputs_postnet_beginning, "b m t -> (b m t)"
        )

        non_tf_estimate_vector_beginning = rearrange(
            non_tf_mel_outputs_postnet_beginning, "b m t -> (b m t)"
        )

        vectors = np.asarray(
            [
                original_vector_beginning,
                tf_estimate_vector_beginning,
                non_tf_estimate_vector_beginning,
            ]
        )
        rho_beginning = np.corrcoef(vectors)

        assert rho_beginning[0, 1] > 0.98
        assert rho_beginning[0, 2] < 0.98

        tf_estimate_vector = rearrange(
            mel_outputs_postnet_tf.detach().numpy(), "b m t -> (b m t)"
        )
        original_vector = rearrange(
            sample_inference_tf_spectrogram.detach().numpy(), "b m t -> (b m t)"
        )

        rho_total = np.corrcoef(
            tf_estimate_vector,
            original_vector,
        )
        assert rho_total[0, 1] > 0.99

        torch.random.manual_seed(1235)
        np.random.seed(1235)
        (
            mel_outputs,
            mel_outputs_postnet_tf,
            gate_outputs,
            alignments,
        ) = lj_speech_tacotron2.inference_partial_tf(
            input_text=sequences,
            input_lengths=input_lengths,
            speaker_ids=None,
            embedded_gst=None,
            tf_mel=sample_inference_spectrogram,
            tf_until_idx=tf_index,
        )
        assert rho_beginning[0, 1] > 0.98

        tf_estimate_vector_beginning = rearrange(
            tf_mel_outputs_postnet_beginning, "b m t -> (b m t)"
        )

        vectors = np.asarray([original_vector_beginning, tf_estimate_vector_beginning])
        rho_beginning = np.corrcoef(vectors)
        assert rho_beginning[0, 1] > 0.98
