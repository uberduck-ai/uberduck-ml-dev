# TODO (Sam): unite the 5 different forward / inference methods in the decoder as well.
# TODO (Sam): treat "gst" and "speaker_embedding" generically (e.g. x_encoding, y_encoding)
# TODO (Sam): move to Hydra or more organized config
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
from typing import Optional

from speechbrain.pretrained import EncoderClassifier

from ..vendor.tfcompat.hparam import HParams
from ..utils.utils import get_mask_from_lengths
from .base import TTSModel
from ..vendor.tfcompat.hparam import HParams
from .base import DEFAULTS as MODEL_DEFAULTS
from .components.decoders.tacotron2 import Decoder
from .components.encoders.tacotron2 import Encoder
from .components.postnet import Postnet
from ..text.symbols import NVIDIA_TACO2_SYMBOLS
from .common import WIN_LENGTH, HOP_LENGTH, SAMPLING_RATE, FILTER_LENGTH, N_MEL_CHANNELS
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

SPEAKER_ENCODER = "speaker_encoder"
TORCHMOJI_ENCODER = "torchmoji_encoder"
AUDIO_ENCODER = "audio_encoder"
GLOBAL_ENCODERS = [SPEAKER_ENCODER, TORCHMOJI_ENCODER, AUDIO_ENCODER]
ENGLISH_CLEANERS = "english_cleaners"

DEFAULTS = HParams(
    fp16_run=False,
    # Text parameters
    symbols_embedding_dim=512,
    mask_padding=True,
    # encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,
    # decoder parameters
    coarse_n_frames_per_step=None,
    decoder_rnn_dim=1024,
    prenet_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,
    p_teacher_forcing=1.0,
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
    # reference encoder
    ref_enc_filters=[32, 32, 64, 64, 128, 128],
    ref_enc_size=[3, 3],
    ref_enc_strides=[2, 2],
    ref_enc_pad=[1, 1],
    # Audio parameters
    ref_enc_gru_size=128,
    num_heads=8,
    sampling_rate=SAMPLING_RATE,
    n_mel_channels=N_MEL_CHANNELS,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    filter_length=FILTER_LENGTH,
    checkpoint_name=None,
    max_wav_value=32768.0,
    mel_fmax=8000,
    mel_fmin=0,
    n_frames_per_step_initial=1,
    win_length=1024,
    # TODO (Sam): Treat all "GSTs" (emotion, speaker, quality) generically.  Rename.
    # TODO (Sam): Need heirarchical defaulting structure so that this is listed as a default param if gst_type is not None
    gst_type=None,
    with_gst=False,
    gst_dim=2304,
    # f0 parameters
    with_f0=False,
    # Speaker encoder parameters
    n_speakers=1,
    speaker_embedding_dim=128,
    audio_encoder_dim=192,
    with_audio_encoding=False,
    audio_encoder_path=None,
    has_speaker_embedding=False,
    # Text parameters
    symbol_set=NVIDIA_TACO2_SYMBOLS,  # should this be here?
    text_cleaners=[ENGLISH_CLEANERS],
)

config = DEFAULTS.values()
config.update(MODEL_DEFAULTS.values())
DEFAULTS = HParams(**config)


class Tacotron2(TTSModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run  # TODO (Sam): remove
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
        # TODO (Sam): make these names match
        self.has_speaker_embedding = hparams.has_speaker_embedding
        self.with_gst = hparams.with_gst

        self.cudnn_enabled = hparams.cudnn_enabled

        if self.n_speakers > 1 and not self.has_speaker_embedding:
            raise Exception("Speaker embedding is required if n_speakers > 1")
        if hparams.has_speaker_embedding:
            self.speaker_embedding = nn.Embedding(
                self.n_speakers, hparams.speaker_embedding_dim
            )
        else:
            self.speaker_embedding = None

        if self.has_speaker_embedding:
            #   NOTE (Sam): self.spkr_lin = ZeroNetwork() if n_speakers == 1 gives fewer trainable terms, and could potentially be better for fine tuning, although we don't really know.
            self.spkr_lin = nn.Linear(
                self.speaker_embedding_dim, self.encoder_embedding_dim
            )

        self.gst_init(hparams)
        self.audio_encoder_init(hparams)

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

    def audio_encoder_init(self, hparams):
        self.audio_encoder = None
        if hparams.audio_encoder_path:

            self.audio_encoder_lin = nn.Linear(
                hparams.audio_encoder_dim, hparams.encoder_embedding_dim
            )
            self.audio_encoder = EncoderClassifier.from_hparams(
                source=hparams.audio_encoder_path
            )
            print("Initialized Audio Encoder")

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

    # NOTE (Sam): it is unclear whether forward should take encoder outputs as arguements or compute them.
    def forward(
        self,
        input_text,
        input_lengths,
        speaker_ids,
        mode=TEACHER_FORCED,
        # TODO (Sam): treat all global encodings the same way.
        embedded_gst: Optional[torch.tensor] = None,
        # NOTE (Sam): can have an audio_encoding of speaker by taking mean audio_encoding.
        audio_encoding: Optional[torch.tensor] = None,
        targets: Optional[torch.tensor] = None,
        output_lengths: Optional[torch.tensor] = None,
        attention: Optional[torch.tensor] = None,
        # TODO (Sam): use inference_double_tf for inference, forward, left_tf, and double_tf.
        # NOTE (Sam): [0, mel_stop_index) tf, (mel_stop_index, mel_start_index) inf, (mel_start_index, max) tf
        mel_start_index: Optional[int] = 0,
        mel_stop_index: Optional[int] = 0,
    ):

        if speaker_ids is not None:
            if max(speaker_ids) >= self.n_speakers:
                raise Exception("Speaker id out of range")
        if input_lengths is not None:
            input_lengths = input_lengths.data
        if output_lengths is not None:
            output_lengths = output_lengths.data

        embedded_inputs = self.embedding(input_text).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        # NOTE (Sam): in a previous version, has_speaker_embedding was implicitly set to be false for n_speakers = 1.
        if self.has_speaker_embedding is True:
            if self.audio_encoder is not None:
                # NOTE (Sam): right now, audio_encoding is a mean of the audio encoder outputs and only works for a single speaker.
                encoder_outputs += self.audio_encoder_lin(audio_encoding)
            else:
                # NOTE (Sam): its unclear where speaker_embedding adds a useful degree of freedom for training.
                # It seems we could use a deeper embedding of the pre-trained encoding to get the same effect.
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
