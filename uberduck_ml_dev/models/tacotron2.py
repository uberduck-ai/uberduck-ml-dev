from torch import nn
import numpy as np
import torch
from torch.nn import functional as F

# NOTE (Sam): components is redundant with common - we needed a different name to make the package imports work as we migrate to a folder based approach
from ..vendor.tfcompat.hparam import HParams
from ..utils.utils import get_mask_from_lengths
from ..data.batch import Batch
from .base import TTSModel
from ..vendor.tfcompat.hparam import HParams
from .base import DEFAULTS as MODEL_DEFAULTS
from .components.decoders.tacotron2 import Decoder, DecoderForwardIsInfer
from .components.encoders.tacotron2 import Encoder, EncoderForwardIsInfer
from .components.postnet import Postnet
from .components.zero_network import ZeroNetwork

DEFAULTS = HParams(
    symbols_embedding_dim=512,
    fp16_run=False,
    mask_padding=True,
    n_mel_channels=80,
    # encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,
    # decoder parameters
    coarse_n_frames_per_step=None,
    decoder_rnn_dim=1024,
    prenet_dim=256,
    prenet_f0_n_layers=1,
    prenet_f0_dim=1,
    prenet_f0_kernel_size=1,
    prenet_rms_dim=0,
    prenet_fms_kernel_size=1,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,
    p_teacher_forcing=1.0,
    pos_weight=None,
    # attention parameters
    attention_rnn_dim=1024,
    attention_dim=128,
    # location layer parameters
    attention_location_n_filters=32,
    attention_location_kernel_size=31,
    # mel post-processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,
    n_speakers=1,
    speaker_embedding_dim=128,
    # reference encoder
    ref_enc_filters=[32, 32, 64, 64, 128, 128],
    ref_enc_size=[3, 3],
    ref_enc_strides=[2, 2],
    ref_enc_pad=[1, 1],
    with_gst=False,
    has_speaker_embedding=False,
    filter_length=1024,
    hop_length=256,
    include_f0=False,
    ref_enc_gru_size=128,
    symbol_set="nvidia_taco2",
    num_heads=8,
    text_cleaners=["english_cleaners"],
    sampling_rate=22050,
    checkpoint_name=None,
    max_wav_value=32768.0,
    mel_fmax=8000,
    mel_fmin=0,
    n_frames_per_step_initial=1,
    win_length=1024,
    gst_type=None,
    torchmoji_model_file=None,
    torchmoji_vocabulary_file=None,
    sample_inference_speaker_ids=None,
    sample_inference_text="That quick beige fox jumped in the air loudly over the thin dog fence.",
    distributed_run=False,
)

config = DEFAULTS.values()
config.update(MODEL_DEFAULTS.values())
DEFAULTS = HParams(**config)


class Tacotron2(TTSModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.pos_weight = hparams.pos_weight
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step_initial = hparams.n_frames_per_step_initial
        self.n_frames_per_step_current = hparams.n_frames_per_step_initial
        self.embedding = nn.Embedding(self.n_symbols, hparams.symbols_embedding_dim)
        std = np.sqrt(2.0 / (self.n_symbols + hparams.symbols_embedding_dim))
        val = np.sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.has_speaker_embedding = hparams.has_speaker_embedding
        self.cudnn_enabled = hparams.cudnn_enabled

        if self.n_speakers > 1 and not self.has_speaker_embedding:
            raise Exception("Speaker embedding is required if n_speakers > 1")
        if hparams.has_speaker_embedding:
            self.speaker_embedding = nn.Embedding(
                self.n_speakers, hparams.speaker_embedding_dim
            )
        else:
            self.speaker_embedding = None

        # NOTE (Sam): it is not totally clear to me this should be split - what if we want to optimize position within the speaker_embedding_dim?
        # What is the role of has_speaker_embedding?
        if self.n_speakers > 1:
            self.spkr_lin = nn.Linear(
                self.speaker_embedding_dim, self.encoder_embedding_dim
            )
        else:
            self.spkr_lin = ZeroNetwork()

        self.gst_init(hparams)

    def gst_init(self, hparams):
        self.gst_lin = None
        self.gst_type = None

        if hparams.get("gst_type") == "torchmoji":
            assert hparams.gst_dim, "gst_dim must be set"
            self.gst_type = hparams.get("gst_type")
            self.gst_lin = nn.Linear(hparams.gst_dim, self.encoder_embedding_dim)
            print("Initialized Torchmoji GST")
        else:
            print("Not using any style tokens")

    def mask_output(
        self, output_lengths, mel_outputs, mel_outputs_postnet, gate_predicted
    ):

        if self.mask_padding:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = F.pad(mask, (0, mel_outputs.size(2) - mask.size(2)))
            mask = mask.permute(1, 0, 2)  # NOTE (Sam): replace with einops

            mel_outputs.data.masked_fill_(mask, 0.0)
            mel_outputs_postnet.data.masked_fill_(mask, 0.0)
            gate_predicted.data.masked_fill_(mask[:, 0, :], 1e3)

        return output_lengths, mel_outputs, mel_outputs_postnet, gate_predicted

    @torch.no_grad()
    def get_alignment(self, inputs):
        (
            input_text,
            input_lengths,
            targets,
            max_len,
            output_lengths,
            speaker_ids,
            *_,
        ) = inputs

        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        if self.speaker_embedding is not None:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            # NOTE (Sam): this requires more careful thought.
            if self.n_speakers > 1:
                encoder_outputs += self.spkr_lin(embedded_speakers)

        encoder_outputs = torch.cat((encoder_outputs,), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )
        return alignments

    def forward(
        self,
        input_text,
        input_lengths,
        targets,
        output_lengths,
        speaker_ids,
        embedded_gst,
    ):

        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        if self.speaker_embedding is not None:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            # NOTE (Sam): this requires more careful thought
            if self.n_speakers > 1:
                encoder_outputs += self.spkr_lin(embedded_speakers)

        if self.gst_lin is not None:
            assert (
                embedded_gst is not None
            ), f"embedded_gst is None but gst_type was set to {self.gst_type}"
            encoder_outputs += self.gst_lin(embedded_gst)
        mel_outputs, gate_predicted, alignments = self.decoder(
            memory=encoder_outputs,
            decoder_inputs=targets,
            memory_lengths=input_lengths,
        )
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        (
            output_lengths,
            mel_outputs,
            mel_outputs_postnet,
            gate_predicted,
        ) = self.mask_output(
            mel_outputs=mel_outputs,
            mel_outputs_postnet=mel_outputs_postnet,
            gate_predicted=gate_predicted,
            output_lengths=output_lengths,
        )

        # NOTE (Sam): batch class in inference methods breaks torchscript
        output = dict(
            mel_outputs=mel_outputs,
            mel_outputs_postnet=mel_outputs_postnet,
            gate_predicted=gate_predicted,
            alignments=alignments,
            output_lengths=output_lengths,
        )
        return output

    @torch.no_grad()
    def inference(self, input_text, input_lengths, speaker_ids, embedded_gst):

        # NOTE (Sam): could compute input_lengths = torch.LongTensor([utterance.shape[1]]) here.
        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        if self.speaker_embedding is not None:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            # NOTE (Sam): this requires more careful thought
            if self.n_speakers > 1:
                encoder_outputs += self.spkr_lin(embedded_speakers)

        if self.gst_lin is not None:
            assert (
                embedded_gst is not None
            ), f"embedded_gst is None but gst_type was set to {self.gst_type}"
            encoder_outputs += self.gst_lin(embedded_gst)

        mel_outputs, gate_predicted, alignments, mel_lengths = self.decoder.inference(
            encoder_outputs, input_lengths
        )
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        (
            output_lengths,
            mel_outputs,
            mel_outputs_postnet,
            gate_predicted,
        ) = self.mask_output(
            mel_outputs=mel_outputs,
            mel_outputs_postnet=mel_outputs_postnet,
            gate_predicted=gate_predicted,
            output_lengths=mel_lengths,
        )

        # NOTE (Sam): batch class in inference methods breaks torchscript
        output = dict(
            mel_outputs=mel_outputs,
            mel_outputs_postnet=mel_outputs_postnet,
            gate_predicted=gate_predicted,
            alignments=alignments,
            output_lengths=output_lengths,
        )
        return output

    def inference_double_tf(
        self,
        input_text,
        input_lengths,
        speaker_ids,
        embedded_gst,
        mel_template,
        mel_start_index,
        mel_stop_index,
    ):
        # NOTE (Sam): forward, inference_partial_tf, and inference are special cases of this function so they should be removed.
        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        if self.speaker_embedding is not None:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            # NOTE (Sam): this requires more careful thought
            if self.n_speakers > 1:
                encoder_outputs += self.spkr_lin(embedded_speakers)

        if self.gst_lin is not None:
            assert (
                embedded_gst is not None
            ), f"embedded_gst is None but gst_type was set to {self.gst_type}"
            encoder_outputs += self.gst_lin(embedded_gst)

        output = self.inference_double_tf(
            self=self.decoder,
            memory=encoder_outputs,
            decoder_inputs=mel_template,
            memory_lengths=input_lengths,
            mel_start_index=mel_start_index,
            mel_stop_index=mel_stop_index,
        )

        return output

    @torch.no_grad()
    def inference_partial_tf(
        self,
        input_text,
        input_lengths,
        speaker_ids,
        embedded_gst,
        tf_mel,
        tf_until_idx,
        device="cpu",
    ):
        """Run inference with partial teacher forcing.

        Teacher forcing is done until tf_until_idx in the mel spectrogram.
        Make sure you pass the mel index and not the text index!

        tf_mel: (B, T, n_mel_channels)
        """
        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        if self.speaker_embedding:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            if self.n_speakers > 1:
                encoder_outputs += self.spkr_lin(embedded_speakers)

        if self.gst_lin is not None:
            assert (
                embedded_gst is not None
            ), f"embedded_gst is None but gst_type was set to {self.gst_type}"
            encoder_outputs += self.gst_lin(embedded_gst)

        mel_outputs, gate_predicted, alignments = self.decoder.inference_partial_tf(
            memory=encoder_outputs,
            decoder_inputs=tf_mel,
            tf_until_idx=tf_until_idx,
            device=device,
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # NOTE (Sam): no mask_output call since output_lengths is not given or predicted
        output = Batch(
            mel_outputs=mel_outputs,
            mel_outputs_postnet=mel_outputs_postnet,
            gate_predicted=gate_predicted,
            alignments=alignments,
        )

        return output


class Tacotron2ForwardIsInfer(Tacotron2):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.encoder = EncoderForwardIsInfer(hparams)
        self.decoder = DecoderForwardIsInfer(hparams)

    def forward(
        self,
        input_text,
        input_lengths,
        speaker_ids,
        embedded_gst,
    ):
        return self.inference(input_text, input_lengths, speaker_ids, embedded_gst)
