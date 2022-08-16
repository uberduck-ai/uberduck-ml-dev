from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS
from uberduck_ml_dev.models.tacotron2 import Tacotron2
from uberduck_ml_dev.trainer.tacotron2 import Tacotron2Trainer
import json
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.common import MelSTFT
from einops import rearrange
import torch
import numpy as np
import random


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

        X, y = model.parse_batch(batch)
        forward_output = model(X)
        assert len(forward_output) == 4

    def test_teacher_forced_inference(self, lj_speech_tacotron2):

        inference = lj_speech_tacotron2.inference_partial_tf

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
        speaker_ids = None
        input_lengths = input_lengths.repeat(1, nreps).squeeze(0)
        input_ = sequences, input_lengths, speaker_ids, None
        (
            mel_outputs,
            mel_outputs_postnet,
            gate_outputs,
            alignments,
            mel_lengths,
        ) = lj_speech_tacotron2.inference(input_)

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

    def test_tf_inference(
        self,
        lj_speech_tacotron2,
        sample_inference_spectrogram,
        sample_tf_inference_spectrogram,
    ):

        torch.random.manual_seed(1234)
        np.random.seed(1234)
        tf_index = 111
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
        speaker_ids = None
        input_lengths = input_lengths.repeat(1, nreps).squeeze(0)
        input_ = sequences, input_lengths, speaker_ids, None
        (
            mel_outputs,
            mel_outputs_postnet_tf,
            gate_outputs,
            alignments,
            mel_lengths,
        ) = lj_speech_tacotron2.inference_partial_tf(
            input_, sample_inference_spectrogram, tf_index
        )

        (
            mel_outputs,
            mel_outputs_postnet_original,
            gate_outputs,
            alignments,
            mel_lengths,
        ) = lj_speech_tacotron2.inference(input_)

        non_tf_mel_outputs_postnet_beginning = (
            mel_outputs_postnet_original.detach().numpy()[:, :, :tf_index]
        )

        tf_mel_outputs_postnet_beginning = mel_outputs_postnet_tf.detach().numpy()[
            :, :, :tf_index
        ]
        sample_inference_spectrogram_beginning = (
            sample_inference_spectrogram.detach().numpy()[:, :, :tf_index]
        )

        original_vector = rearrange(
            sample_inference_spectrogram_beginning.detach().numpy(), "b m t -> (b m t)"
        )

        tf_estimate_vector = rearrange(
            tf_mel_outputs_postnet_beginning.detach().numpy(), "b m t -> (b m t)"
        )

        non_tf_estimate_vector = rearrange(
            non_tf_mel_outputs_postnet_beginning.detach().numpy(), "b m t -> (b m t)"
        )

        rho_beginning = np.corrcoef(
            original_vector, tf_estimate_vector, non_tf_estimate_vector
        )

        assert rho_beginning[0, 1] > 0.99
        assert rho_beginning[0, 2] < 0.99

        tf_estimate_vector = rearrange(
            mel_outputs_postnet_tf.detach().numpy(), "b m t -> (b m t)"
        )
        original_vector = rearrange(
            sample_inference_spectrogram.detach().numpy(), "m t -> (m t)"
        )

        rho_total = np.corrcoef(
            tf_estimate_vector,
            original_vector,
        )[0, 1]
        assert rho_total[0, 1] > 0.99
