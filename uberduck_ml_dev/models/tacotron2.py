# TODO (Sam): unite the 5 different forward / inference methods in the decoder as well.
# TODO (Sam): treat "gst" and "speaker_embedding" generically (e.g. x_encoding, y_encoding)
# TODO (Sam): move to Hydra or more organized config
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
from typing import Optional

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

TEACHER_FORCED = "teacher-forced"
LEFT_TEACHER_FORCED = "left-teacher-forced"
DOUBLE_TEACHER_FORCED = "double-teacher-forced"
INFERENCE = "inference"
ATTENTION_FORCED = "attention-forced"
MODES = [
    TEACHER_FORCED,
    LEFT_TEACHER_FORCED,
    DOUBLE_TEACHER_FORCED,
    INFERENCE,
    ATTENTION_FORCED,
]

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
    # TODO (Sam): Treat all "GSTs" (emotion, speaker, quality) generically.  Rename
    gst_type=None,
    with_gst=False,
    gst_dim=2304,  # Need heirarchical defaulting structure so that this is listed as a default param if gst_type is not None
    torchmoji_model_file=None,
    torchmoji_vocabulary_file=None,
    # NOTE (Sam): to-do - move sample_inference parameters to trainer.
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
        self.with_gst = hparams.with_gst

        if self.n_speakers > 1 and not self.has_speaker_embedding:
            raise Exception("Speaker embedding is required if n_speakers > 1")
        if hparams.has_speaker_embedding:
            if self.new_speaker_embedding:
                self.speaker_embedding = nn.Embedding(
                    self.n_speakers, hparams.speaker_embedding_dim
                )
            # NOTE (Sam): treating context encoders generically will remove such code.
            if self.speechbrain_speaker_embedding:
                self.speaker_embedding = self.mean_embedding_function()
        else:
            self.speaker_embedding = None

        if self.has_speaker_embedding:
            #   NOTE (Sam): self.spkr_lin = ZeroNetwork() if n_speakers == 1 gives fewer trainable terms, and could potentially be better for fine tuning, although we don't really know.
            self.spkr_lin = nn.Linear(
                self.speaker_embedding_dim, self.encoder_embedding_dim
            )

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

    def forward(
        self,
        input_text,
        input_lengths,
        speaker_ids,
        mode=TEACHER_FORCED,
        # TODO (Sam): rename "emotional_encoding"
        embedded_gst: Optional[torch.tensor] = None,
        # NOTE (Sam): can have an audio_encoding of speaker by taking mean audio_encoding.
        audio_encoding: Optional[torch.tensor] = None,
        targets: Optional[torch.tensor] = None,
        output_lengths: Optional[torch.tensor] = None,
        attention: Optional[torch.tensor] = None,
        # TODO (Sam): use these to set inference, forward, left_tf, and double_tf as the same mode.
        # NOTE (Sam): [0, mel_stop_index) tf, (mel_stop_index, mel_start_index) inf, (mel_start_index, max) tf
        mel_start_index: Optional[int] = 0,
        mel_stop_index: Optional[int] = 0,
    ):

        if input_lengths is not None:
            input_lengths = input_lengths.data
        if output_lengths is not None:
            output_lengths = output_lengths.data

        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        # NOTE (Sam): in a previous version, has_speaker_embedding was implicitly set to be false for n_speakers = 1.
        if self.has_speaker_embedding is True:
            if self.has_audio_embedding is True:
                embedded_speakers += self.audio_lin(audio_encoding)
            else:
                embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
                encoder_outputs += self.spkr_lin(embedded_speakers)

        if self.with_gst:
            assert (
                embedded_gst is not None
            ), f"embedded_gst is None but gst_type was set to {self.gst_type}"
            encoder_outputs += self.gst_lin(embedded_gst)

        if mode == TEACHER_FORCED:
            mel_outputs, gate_predicted, alignments = self.decoder(
                memory=encoder_outputs,
                decoder_inputs=targets,
                memory_lengths=input_lengths,
            )

        if mode == INFERENCE:
            (
                mel_outputs,
                gate_predicted,
                alignments,
                output_lengths,
            ) = self.decoder.inference(encoder_outputs, input_lengths)

        if mode == DOUBLE_TEACHER_FORCED:
            # TODO (Sam): use inference_double_tf for inference, forward, left_tf, and double_tf
            mel_outputs, gate_predicted, alignments = self.decoder.inference_double_tf(
                memory=encoder_outputs,
                decoder_inputs=targets,
                memory_lengths=input_lengths,
                mel_start_index=mel_start_index,
                mel_stop_index=mel_stop_index,
            )

        if mode == LEFT_TEACHER_FORCED:
            mel_outputs, gate_predicted, alignments = self.decoder.inference_partial_tf(
                memory=encoder_outputs,
                decoder_inputs=targets,
                tf_until_idx=mel_stop_index,
            )

        if mode == ATTENTION_FORCED:
            # NOTE (Sam): decoder.inference_noattention does not return output lengths.
            output_lengths = (
                torch.LongTensor([0])
                if not self.cudnn_enabled
                else torch.LongTensor([0]).cuda()
            )
            (
                mel_outputs,
                gate_predicted,
                alignments,
            ) = self.decoder.inference_noattention(encoder_outputs, attention)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if mode in [INFERENCE, TEACHER_FORCED, ATTENTION_FORCED]:
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

        # NOTE (Sam): batch class in inference methods breaks torchscript.
        output = dict(
            mel_outputs=mel_outputs,
            mel_outputs_postnet=mel_outputs_postnet,
            gate_predicted=gate_predicted,
            alignments=alignments,
            output_lengths=output_lengths,
        )
        return output


# NOTE (Sam): I'm not sure if this is necessary anymore since inference is now in forward.
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
