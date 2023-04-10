from torch import nn
import torch
from torch.nn import functional as F
from typing import Optional, List
from torch.cuda.amp import autocast
import numpy as np

from ...common import LinearNorm
from ..attention import Attention
from ..prenet import Prenet
from ....utils.utils import get_mask_from_lengths
from einops import rearrange


class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step_initial = hparams.n_frames_per_step_initial
        self.n_frames_per_step_current = hparams.n_frames_per_step_initial
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.cudnn_enabled = hparams.cudnn_enabled
        self.attention_hidden = torch.tensor([])
        self.attention_cell = torch.tensor([])
        self.decoder_hidden = torch.tensor([])
        self.decoder_cell = torch.tensor([])
        self.attention_weights = torch.tensor([])
        self.attention_weights_cum = torch.tensor([])
        self.attention_context = torch.tensor([])
        self.memory = torch.tensor([])
        self.processed_memory = torch.tensor([])
        self.mask = torch.tensor([])

        self.prenet = Prenet(
            hparams.n_mel_channels,
            [hparams.prenet_dim, hparams.prenet_dim],
        )

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim,
        )

        self.attention_layer = Attention(
            hparams.attention_rnn_dim,
            self.encoder_embedding_dim,
            hparams.attention_dim,
            hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size,
            fp16_run=hparams.fp16_run,
        )

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim,
            1,
        )

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step_initial,
        )

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

    def set_current_frames_per_step(self, n_frames: int):
        self.n_frames_per_step_current = n_frames

    def get_go_frame(self, memory):
        """Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = memory.data.new_zeros(B, self.n_mel_channels)
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = memory.data.new_zeros(B, self.attention_rnn_dim)
        self.attention_cell = memory.data.new_zeros(B, self.attention_rnn_dim)

        self.decoder_hidden = memory.data.new_zeros(B, self.decoder_rnn_dim)
        self.decoder_cell = memory.data.new_zeros(B, self.decoder_rnn_dim)
        self.attention_weights = memory.data.new_zeros(B, MAX_TIME)
        self.attention_weights_cum = memory.data.new_zeros(B, MAX_TIME)
        self.attention_context = memory.data.new_zeros(B, self.encoder_embedding_dim)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    #   NOTE (Sam): spaghetti code - comment this out after removing dependency in partial_tf index
    def parse_decoder_inputs(self, decoder_inputs):
        """Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)

        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.contiguous()
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step_current),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(
        self,
        mel_outputs,
        gate_outputs: List[torch.Tensor],
        alignments: List[torch.Tensor],
    ):
        """Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) > 1:
            gate_outputs = gate_outputs.transpose(0, 1)
        else:
            gate_outputs = gate_outputs[None]
        gate_outputs = gate_outputs.contiguous()
        # (B, T_out, n_mel_channels * n_frames_per_step) -> (B, T_out * n_frames_per_step, n_mel_channels)
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_weights: Optional[torch.Tensor]):
        """Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training
        )
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
            attention_weights,
        )

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training
        )
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )

        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        B = memory.size(0)
        decoder_inputs = rearrange(
            decoder_inputs, "b m t -> t b m"
        )  # n_frames_per_step not used anymore
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )

        mel_outputs = torch.empty(
            B, 0, self.n_frames_per_step_current * self.n_mel_channels
        )
        if self.cudnn_enabled:
            mel_outputs = mel_outputs.cuda()
        gate_outputs, alignments = [], []
        desired_output_frames = decoder_inputs.size(0) / self.n_frames_per_step_current
        while mel_outputs.size(1) < desired_output_frames - 1:
            if (
                mel_outputs.size(1) == 0
                or np.random.uniform(0.0, 1.0) <= self.p_teacher_forcing
            ):
                teacher_forced_frame = decoder_inputs[
                    mel_outputs.size(1) * self.n_frames_per_step_current
                ]

                decoder_input = teacher_forced_frame
            else:
                # NOTE(zach): we may need to concat these as we go to ensure that
                # it's easy to retrieve the last n_frames_per_step_init frames.
                to_concat = (
                    self.prenet(
                        mel_outputs[:, -1, -1 * self.n_frames_per_step_current :]
                    ),
                )
                decoder_input = torch.cat(to_concat, dim=1)
            # NOTE(zach): When training with fp16_run == True, decoder_rnn seems to run into
            # issues with NaNs in gradient, maybe due to vanishing gradients.
            # Disable half-precision for this call to work around the issue.
            with autocast(enabled=False):
                mel_output, gate_output, attention_weights = self.decode(
                    decoder_input, None
                )
            mel_outputs = torch.cat(
                [
                    mel_outputs,
                    mel_output[
                        :, 0 : self.n_mel_channels * self.n_frames_per_step_current
                    ].unsqueeze(1),
                ],
                dim=1,
            )
            gate_outputs += [gate_output.squeeze()] * self.n_frames_per_step_current
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, memory_lengths):
        """Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )

        B = memory.size(0)
        mel_outputs = torch.empty(
            B, 0, self.n_frames_per_step_current * self.n_mel_channels
        )
        if self.cudnn_enabled:
            mel_outputs = mel_outputs.cuda()
        gate_outputs, alignments = [], []

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )

        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input, None)
            mel_output = mel_output[
                :, 0 : self.n_mel_channels * self.n_frames_per_step_current
            ].unsqueeze(1)

            mel_outputs = torch.cat([mel_outputs, mel_output], dim=1)
            gate_outputs += [gate_output.squeeze()] * self.n_frames_per_step_current
            alignments += [alignment]

            dec = (
                torch.le(torch.sigmoid(gate_output), self.gate_threshold)
                .to(torch.int32)
                .squeeze(1)
            )

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if torch.sum(not_finished) == 0:
                break
            if mel_outputs.shape[1] == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output[:, -1, -1 * self.n_mel_channels :]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments, mel_lengths

    def inference_noattention(self, memory, attention_map):
        """Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        B = memory.size(0)
        mel_outputs = torch.empty(
            B, 0, self.n_frames_per_step_current * self.n_mel_channels
        )
        if self.cudnn_enabled:
            mel_outputs = mel_outputs.cuda()
        gate_outputs, alignments = [], []
        for i in range(len(attention_map)):
            attention = attention_map[i]
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input, attention)
            mel_output = mel_output[
                :, 0 : self.n_mel_channels * self.n_frames_per_step_current
            ].unsqueeze(1)

            mel_outputs = torch.cat([mel_outputs, mel_output], dim=1)
            gate_outputs += [gate_output.squeeze()] * self.n_frames_per_step_current
            alignments += [alignment]

            decoder_input = mel_output[:, -1, -1 * self.n_mel_channels :]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference_double_tf(
        self, memory, decoder_inputs, memory_lengths, mel_start_index, mel_stop_index
    ):
        # NOTE (Sam): forward, inference_partial_tf, and inference are special cases of this function so they should be removed.
        decoder_inputs = rearrange(decoder_inputs, "b m t -> t b m")
        decoder_input = self.get_go_frame(memory).unsqueeze(0)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )

        mel_outputs, gate_outputs, alignments = [], [], []

        i = 0
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if i >= mel_start_index and i < mel_stop_index:
                decoder_input = self.prenet(mel_outputs[len(mel_outputs) - 1])
            else:
                decoder_input = self.prenet(decoder_inputs[len(mel_outputs), :, :])
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input, None
            )
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
            i += 1

        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        mel_outputs = mel_outputs.transpose(1, 2)

        # NOTE (Sam): consider using parse_decoder_outputs
        return mel_outputs, torch.vstack(gate_outputs), torch.vstack(alignments)

    def inference_partial_tf(self, memory, decoder_inputs, tf_until_idx, device="cpu"):
        """Decoder inference with teacher-forcing up until tf_until_idx
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        if device == "cuda" and self.cudnn_enabled:
            decoder_inputs = decoder_inputs.cuda()

        B = memory.size(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = decoder_inputs.reshape(
            -1, decoder_inputs.size(1), self.n_mel_channels
        )
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs = torch.empty(
            B, 0, self.n_frames_per_step_current * self.n_mel_channels
        )
        if device == "cuda" and self.cudnn_enabled:
            mel_outputs = mel_outputs.cuda()
        gate_outputs, alignments = [], []

        while True:
            if mel_outputs.size(1) < tf_until_idx:
                teacher_forced_frame = decoder_inputs[
                    mel_outputs.size(1) * self.n_frames_per_step_current
                ]

                decoder_input = teacher_forced_frame
            else:
                decoder_input = self.prenet(
                    mel_outputs[:, -1, -1 * self.n_mel_channels :]
                )
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input, None
            )
            mel_outputs = torch.cat(
                [
                    mel_outputs,
                    mel_output[
                        :, 0 : self.n_mel_channels * self.n_frames_per_step_current
                    ].unsqueeze(1),
                ],
                dim=1,
            )
            gate_outputs += [gate_output.squeeze()] * self.n_frames_per_step_current
            alignments += [attention_weights]
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif mel_outputs.size(1) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments


class DecoderForwardIsInfer(Decoder):
    def forward(self, memory, memory_lengths):
        return self.inference(memory, memory_lengths)
