# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torch
from torch import nn
from .common import Encoder, LengthRegulator, ConvAttention, Invertible1x1ConvLUS, Invertible1x1Conv, AffineTransformationLayer, LinearNorm, ExponentialClass
# from common import get_mask_from_lengths
from ..utils.utils import get_mask_from_lengths
from .components.attribute_prediction_model import get_attribute_prediction_model
from .components.alignment import mas_width1 as mas


class FlowStep(nn.Module):
    def __init__(self,  n_mel_channels, n_context_dim, n_layers,
                 affine_model='simple_conv', scaling_fn='exp',
                 matrix_decomposition='', affine_activation='softplus',
                 use_partial_padding=False, cache_inverse=False):
        super(FlowStep, self).__init__()
        if matrix_decomposition == 'LUS':
            self.invtbl_conv = Invertible1x1ConvLUS(n_mel_channels, cache_inverse=cache_inverse)
        else:
            self.invtbl_conv = Invertible1x1Conv(n_mel_channels, cache_inverse=cache_inverse)

        self.affine_tfn = AffineTransformationLayer(
            n_mel_channels, n_context_dim, n_layers, affine_model=affine_model,
            scaling_fn=scaling_fn, affine_activation=affine_activation,
            use_partial_padding=use_partial_padding)

    def enable_inverse_cache(self):
        self.invtbl_conv.cache_inverse=True


    def forward(self, z, context, inverse=False, seq_lens=None):
        if inverse:  # for inference z-> mel
            z = self.affine_tfn(z, context, inverse, seq_lens=seq_lens)
            z = self.invtbl_conv(z, inverse)
            return z
        else:  # training mel->z
            z, log_det_W = self.invtbl_conv(z)
            z, log_s = self.affine_tfn(z, context, seq_lens=seq_lens)
            return z, log_det_W, log_s


# # NOTE (Sam): comment this out for GPU
# def get_mask_from_lengths(lengths):
#     """Constructs binary mask from a 1D torch tensor of input lengths
#     Args:
#         lengths (torch.tensor): 1D tensor
#     Returns:
#         mask (torch.tensor): num_sequences x max_length x 1 binary tensor
#     """
#     max_len = torch.max(lengths).item()
#     ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
#     mask = (ids < lengths.unsqueeze(1)).bool()
#     return mask

from typing import Optional
class RADTTS(torch.nn.Module):
    def __init__(self, n_speakers, n_speaker_dim, n_text, n_text_dim, n_flows,
                 n_conv_layers_per_step, n_mel_channels, n_hidden,
                 mel_encoder_n_hidden, dummy_speaker_embedding, n_early_size,
                 n_early_every, n_group_size, affine_model, dur_model_config,
                 f0_model_config, energy_model_config, v_model_config=None,
                 include_modules='dec', scaling_fn='exp',
                 matrix_decomposition='', learn_alignments=False,
                 affine_activation='softplus', attn_use_CTC=True,
                 use_speaker_emb_for_alignment=False, use_context_lstm=False,
                 context_lstm_norm=None, text_encoder_lstm_norm=None,
                 n_f0_dims=0, n_energy_avg_dims=0,
                 context_lstm_w_f0_and_energy=True,
                 use_first_order_features=False, unvoiced_bias_activation='',
                 ap_pred_log_f0=False, **kwargs):
        super(RADTTS, self).__init__()
        assert(n_early_size % 2 == 0)
        self.do_mel_descaling = kwargs.get('do_mel_descaling', True)
        self.n_mel_channels = n_mel_channels
        self.n_f0_dims = n_f0_dims  # >= 1 to trains with f0
        self.n_energy_avg_dims = n_energy_avg_dims  # >= 1 trains with energy
        self.decoder_use_partial_padding = kwargs.get(
            'decoder_use_partial_padding', True)
        self.n_speaker_dim = n_speaker_dim
        assert(self.n_speaker_dim % 2 == 0)
        self.speaker_embedding = torch.nn.Embedding(
            n_speakers, self.n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = Encoder(encoder_embedding_dim=n_text_dim,
                               norm_fn=nn.InstanceNorm1d,
                               lstm_norm_fn=text_encoder_lstm_norm)
        self.dummy_speaker_embedding = dummy_speaker_embedding
        self.learn_alignments = learn_alignments
        self.affine_activation = affine_activation
        self.include_modules = include_modules
        self.attn_use_CTC = bool(attn_use_CTC)
        self.use_speaker_emb_for_alignment = use_speaker_emb_for_alignment
        self.use_context_lstm = bool(use_context_lstm)
        self.context_lstm_norm = context_lstm_norm
        self.context_lstm_w_f0_and_energy = context_lstm_w_f0_and_energy
        self.length_regulator = LengthRegulator()
        self.use_first_order_features = bool(use_first_order_features)
        self.decoder_use_unvoiced_bias = kwargs.get(
            'decoder_use_unvoiced_bias', True)
        self.ap_pred_log_f0 = ap_pred_log_f0
        self.ap_use_unvoiced_bias = kwargs.get('ap_use_unvoiced_bias', True)
        self.attn_straight_through_estimator = kwargs.get(
            'attn_straight_through_estimator', False)
        if 'atn' in include_modules or 'dec' in include_modules:
            if self.learn_alignments:
                if self.use_speaker_emb_for_alignment:
                    self.attention = ConvAttention(
                        n_mel_channels, n_text_dim + self.n_speaker_dim)
                else:
                    self.attention = ConvAttention(n_mel_channels, n_text_dim)

            self.n_flows = n_flows
            self.n_group_size = n_group_size

            n_flowstep_cond_dims = (
                self.n_speaker_dim +
                (n_text_dim + n_f0_dims + n_energy_avg_dims) * n_group_size)

            if self.use_context_lstm:
                n_in_context_lstm = (
                    self.n_speaker_dim + n_text_dim * n_group_size)
                n_context_lstm_hidden = int(
                    (self.n_speaker_dim + n_text_dim * n_group_size) / 2)

                if self.context_lstm_w_f0_and_energy:
                    n_in_context_lstm = (
                        n_f0_dims + n_energy_avg_dims + n_text_dim)
                    n_in_context_lstm *= n_group_size
                    n_in_context_lstm += self.n_speaker_dim

                    n_context_hidden = (
                        n_f0_dims + n_energy_avg_dims + n_text_dim)
                    n_context_hidden = n_context_hidden * n_group_size / 2
                    n_context_hidden = self.n_speaker_dim + n_context_hidden
                    n_context_hidden = int(n_context_hidden)

                    n_flowstep_cond_dims = (
                        self.n_speaker_dim + n_text_dim * n_group_size)

                self.context_lstm = torch.nn.LSTM(
                    input_size=n_in_context_lstm,
                    hidden_size=n_context_lstm_hidden, num_layers=1,
                    batch_first=True, bidirectional=True)

                if context_lstm_norm is not None:
                    if 'spectral' in context_lstm_norm:
                        print("Applying spectral norm to context encoder LSTM")
                        lstm_norm_fn_pntr = torch.nn.utils.spectral_norm
                    elif 'weight' in context_lstm_norm:
                        print("Applying weight norm to context encoder LSTM")
                        lstm_norm_fn_pntr = torch.nn.utils.weight_norm

                    self.context_lstm = lstm_norm_fn_pntr(
                        self.context_lstm, 'weight_hh_l0')
                    self.context_lstm = lstm_norm_fn_pntr(
                        self.context_lstm, 'weight_hh_l0_reverse')

            if self.n_group_size > 1:
                self.unfold_params = {'kernel_size': (n_group_size, 1),
                                      'stride': n_group_size,
                                      'padding': 0, 'dilation': 1}
                self.unfold = nn.Unfold(**self.unfold_params)

            self.exit_steps = []
            self.n_early_size = n_early_size
            n_mel_channels = n_mel_channels*n_group_size

            for i in range(self.n_flows):
                if i > 0 and i % n_early_every == 0:  # early exitting
                    n_mel_channels -= self.n_early_size
                    self.exit_steps.append(i)

                self.flows.append(FlowStep(
                    n_mel_channels, n_flowstep_cond_dims,
                    n_conv_layers_per_step, affine_model, scaling_fn,
                    matrix_decomposition, affine_activation=affine_activation,
                    use_partial_padding=self.decoder_use_partial_padding))

        if 'dpm' in include_modules:
            dur_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            self.dur_pred_layer = get_attribute_prediction_model(
                dur_model_config)

        self.use_unvoiced_bias = False
        self.use_vpred_module = False
        self.ap_use_voiced_embeddings = kwargs.get(
            'ap_use_voiced_embeddings', True)

        if self.decoder_use_unvoiced_bias or self.ap_use_unvoiced_bias:
            assert (unvoiced_bias_activation in {'relu', 'exp'})
            self.use_unvoiced_bias = True
            if unvoiced_bias_activation == 'relu':
                unvbias_nonlin = nn.ReLU()
            elif unvoiced_bias_activation == 'exp':
                unvbias_nonlin = ExponentialClass()
            else:
                exit(1)  # we won't reach here anyway due to the assertion
            self.unvoiced_bias_module = nn.Sequential(
                LinearNorm(n_text_dim, 1), unvbias_nonlin)

        # all situations in which the vpred module is necessary
        if self.ap_use_voiced_embeddings or self.use_unvoiced_bias or 'vpred' in include_modules:
            self.use_vpred_module = True

        if self.use_vpred_module:
            v_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            self.v_pred_module = get_attribute_prediction_model(v_model_config)
            # 4 embeddings, first two are scales, second two are biases
            if self.ap_use_voiced_embeddings:
                self.v_embeddings = torch.nn.Embedding(4, n_text_dim)

        if 'apm' in include_modules:
            f0_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            energy_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            if self.use_first_order_features:
                f0_model_config['hparams']['n_in_dim'] = 2
                energy_model_config['hparams']['n_in_dim'] = 2
                if 'spline_flow_params' in f0_model_config['hparams'] and f0_model_config['hparams']['spline_flow_params'] is not None:
                    f0_model_config['hparams']['spline_flow_params']['n_in_channels'] = 2
                if 'spline_flow_params' in energy_model_config['hparams'] and energy_model_config['hparams']['spline_flow_params'] is not None:
                    energy_model_config['hparams']['spline_flow_params']['n_in_channels'] = 2
            else:
                if 'spline_flow_params' in f0_model_config['hparams'] and f0_model_config['hparams']['spline_flow_params'] is not None:
                    f0_model_config['hparams']['spline_flow_params']['n_in_channels'] = f0_model_config['hparams']['n_in_dim']
                if 'spline_flow_params' in energy_model_config['hparams'] and energy_model_config['hparams']['spline_flow_params'] is not None:
                    energy_model_config['hparams']['spline_flow_params']['n_in_channels'] = energy_model_config['hparams']['n_in_dim']

            self.f0_pred_module = get_attribute_prediction_model(
                f0_model_config)
            self.energy_pred_module = get_attribute_prediction_model(
                energy_model_config)

    def is_attribute_unconditional(self):
        """
        returns true if the decoder is conditioned on neither energy nor F0
        """
        return self.n_f0_dims == 0 and self.n_energy_avg_dims == 0

    # NOTE (Sam): make this more refined
    def encode_speaker(self, spk_ids: Optional[str], audio_encodings: Optional[torch.Tensor] = None):

        assert not (spk_ids and audio_encodings), "Only one of spk_ids and audio_encodings can be provided"
        if audio_encodings is not None:
            return audio_encodings
        spk_ids = spk_ids * 0 if self.dummy_speaker_embedding else spk_ids
        spk_vecs = self.speaker_embedding(spk_ids)
        return spk_vecs

    def encode_text(self, text, in_lens):
        # text_embeddings: b x len_text x n_text_dim
        text_embeddings = self.embedding(text).transpose(1, 2)
        # text_enc: b x n_text_dim x encoder_dim (512)
        
        if in_lens is None:
            text_enc = self.encoder.infer(text_embeddings).transpose(1, 2)
        else:
            text_enc = self.encoder(text_embeddings, in_lens).transpose(1, 2)

        return text_enc, text_embeddings

    def preprocess_context(self, context, speaker_vecs, out_lens=None, f0=None,
                           energy_avg=None):

        if self.n_group_size > 1:
            # unfolding zero-padded values
            context = self.unfold(context.unsqueeze(-1))
            if f0 is not None:
                f0 = self.unfold(f0[:, None, :, None])
            if energy_avg is not None:
                energy_avg = self.unfold(energy_avg[:, None, :, None])
        speaker_vecs = speaker_vecs[..., None].expand(-1, -1, context.shape[2])
        context_w_spkvec = torch.cat((context, speaker_vecs), 1)

        if self.use_context_lstm:
            if self.context_lstm_w_f0_and_energy:
                if f0 is not None:
                    context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)

                if energy_avg is not None:
                    context_w_spkvec = torch.cat(
                        (context_w_spkvec, energy_avg), 1)

            unfolded_out_lens = (out_lens // self.n_group_size).long().cpu()
            unfolded_out_lens_packed = nn.utils.rnn.pack_padded_sequence(
                context_w_spkvec.transpose(1, 2), unfolded_out_lens,
                batch_first=True, enforce_sorted=False)
            self.context_lstm.flatten_parameters()
            context_lstm_packed_output, _ = self.context_lstm(
                unfolded_out_lens_packed)
            context_lstm_padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                context_lstm_packed_output, batch_first=True)
            context_w_spkvec = context_lstm_padded_output.transpose(1, 2)

        if not self.context_lstm_w_f0_and_energy:
            if f0 is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)

            if energy_avg is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, energy_avg), 1)

        return context_w_spkvec

    def enable_inverse_cache(self):
        for flow_step in self.flows:
            flow_step.enable_inverse_cache()

    def fold(self, mel):
        """Inverse of the self.unfold(mel.unsqueeze(-1)) operation used for the
        grouping or "squeeze" operation on input

        Args:
            mel: B x C x T tensor of temporal data
        """
        mel = nn.functional.fold(
            mel, output_size=(mel.shape[2]*self.n_group_size, 1),
            **self.unfold_params).squeeze(-1)
        return mel

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS. These will
        no longer recieve a gradient
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas(attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    # NOTE (Sam): for cpu compability.
                    # hard_attn, device=attn.get_device())
                    hard_attn, device=attn.device)
        return attn_out

    def get_first_order_features(self, feats, out_lens, dilation=1):
        """
        feats: b x max_length
        out_lens: b-dim
        """
        # add an extra column
        feats_extended_R = torch.cat(
            (feats, torch.zeros_like(feats[:, 0:dilation])), dim=1)
        feats_extended_L = torch.cat(
            (torch.zeros_like(feats[:, 0:dilation]), feats), dim=1)
        dfeats_R = feats_extended_R[:, dilation:] - feats
        dfeats_L = feats - feats_extended_L[:, 0:-dilation]

        return (dfeats_R + dfeats_L) * 0.5

    def apply_voice_mask_to_text(self, text_enc, voiced_mask):
        """
        text_enc: b x C x N
        voiced_mask: b x N
        """

        voiced_mask = voiced_mask.unsqueeze(1)
        voiced_embedding_s = self.v_embeddings.weight[0:1, :, None]
        unvoiced_embedding_s = self.v_embeddings.weight[1:2, :, None]
        voiced_embedding_b = self.v_embeddings.weight[2:3, :, None]
        unvoiced_embedding_b = self.v_embeddings.weight[3:4, :, None]
        scale = torch.sigmoid(voiced_embedding_s*voiced_mask + unvoiced_embedding_s*(1-voiced_mask))
        bias = 0.1*torch.tanh(voiced_embedding_b*voiced_mask + unvoiced_embedding_b*(1-voiced_mask))
        return text_enc*scale+bias

    def forward(self, mel, speaker_ids, text, in_lens, out_lens,
                binarize_attention=False, attn_prior=None,
                f0=None, energy_avg=None, voiced_mask=None, p_voiced=None, audio_embedding = None):
        
        # NOTE (Sam): hacky solution until check speaker_ids isn't being used as a positional argument.
        # encode_speaker can also perform this nullification
        if audio_embedding is not None:
            speaker_ids = None

        if speaker_ids is not None:
            speaker_vecs = self.encode_speaker(speaker_ids)

        if audio_embedding is not None:
            speaker_vecs = audio_embedding

        # print(text.type())
        text_enc, text_embeddings = self.encode_text(text, in_lens)
        # print(text_enc.type(), text_enc)
        # text_enc = text_enc.double() # NOTE (Sam): this was necessary for inference without dataloader - no clue why.
        log_s_list, log_det_W_list, z_mel = [], [], []
        attn = None
        attn_soft = None
        attn_hard = None
        if 'atn' in self.include_modules or 'dec' in self.include_modules:
            # make sure to do the alignments before folding
            attn_mask = get_mask_from_lengths(in_lens)[..., None] == 0

            text_embeddings_for_attn = text_embeddings
            if self.use_speaker_emb_for_alignment:
                speaker_vecs_expd = speaker_vecs[:, :, None].expand(
                    -1, -1, text_embeddings.shape[2])
                text_embeddings_for_attn = torch.cat(
                    (text_embeddings_for_attn, speaker_vecs_expd.detach()), 1)

            # attn_mask shld be 1 for unsd t-steps in text_enc_w_spkvec tensor
            attn_soft, attn_logprob = self.attention(
                mel, text_embeddings_for_attn, out_lens, attn_mask,
                key_lens=in_lens, attn_prior=attn_prior)

            if binarize_attention:
                attn = self.binarize_attention(attn_soft, in_lens, out_lens)
                attn_hard = attn
                if self.attn_straight_through_estimator:
                    attn_hard = attn_soft + (attn_hard - attn_soft).detach()
            else:
                attn = attn_soft

            # print(text_enc.type(),attn.type(), 'report card')
            context = torch.bmm(text_enc, attn.squeeze(1).transpose(1, 2))

        f0_bias = 0
        # unvoiced bias forward pass
        if self.use_unvoiced_bias:
            f0_bias = self.unvoiced_bias_module(context.permute(0, 2, 1))
            f0_bias = -f0_bias[..., 0]
            f0_bias = f0_bias * (~voiced_mask.bool()).float()

        # mel decoder forward pass
        if 'dec' in self.include_modules:
            if self.n_group_size > 1:
                # might truncate some frames at the end, but that's ok
                # sometimes referred to as the "squeeeze" operation
                # invert this by calling self.fold(mel_or_z)
                mel = self.unfold(mel.unsqueeze(-1))
            z_out = []
            # where context is folded
            # mask f0 in case values are interpolated

            if f0 is None:
                f0_aug = None
            else:
                if self.decoder_use_unvoiced_bias:
                    f0_aug = f0 * voiced_mask + f0_bias
                else:
                    f0_aug = f0 * voiced_mask

            context_w_spkvec = self.preprocess_context(
                    context, speaker_vecs, out_lens, f0_aug,
                    energy_avg)

            log_s_list, log_det_W_list, z_out = [], [], []
            unfolded_seq_lens = out_lens//self.n_group_size
            for i, flow_step in enumerate(self.flows):
                if i in self.exit_steps:
                    z = mel[:, :self.n_early_size]
                    z_out.append(z)
                    mel = mel[:, self.n_early_size:]
                mel, log_det_W, log_s = flow_step(
                    mel, context_w_spkvec, seq_lens=unfolded_seq_lens)
                log_s_list.append(log_s)
                log_det_W_list.append(log_det_W)

            z_out.append(mel)
            z_mel = torch.cat(z_out, 1)

        # duration predictor forward pass
        duration_model_outputs = None
        if 'dpm' in self.include_modules:
            if attn_hard is None:
                attn_hard = self.binarize_attention(
                    attn_soft, in_lens, out_lens)

            # convert hard attention to durations
            attn_hard_reduced = attn_hard.sum(2)[:, 0, :]
            duration_model_outputs = self.dur_pred_layer(
                torch.detach(text_enc),
                torch.detach(speaker_vecs),
                torch.detach(attn_hard_reduced.float()), in_lens)

        # f0, energy, vpred predictors forward pass
        f0_model_outputs = None
        energy_model_outputs = None
        vpred_model_outputs = None
        if 'apm' in self.include_modules:
            if attn_hard is None:
                attn_hard = self.binarize_attention(
                    attn_soft, in_lens, out_lens)

            # convert hard attention to durations
            if binarize_attention:
                text_enc_time_expanded = context.clone()
            else:
                text_enc_time_expanded = torch.bmm(
                    text_enc, attn_hard.squeeze(1).transpose(1, 2))

            if self.use_vpred_module:
                # unvoiced bias requires voiced mask prediction
                vpred_model_outputs = self.v_pred_module(
                    torch.detach(text_enc_time_expanded),
                    torch.detach(speaker_vecs),
                    torch.detach(voiced_mask), out_lens)

                # affine transform context using voiced mask
                if self.ap_use_voiced_embeddings:
                    text_enc_time_expanded = self.apply_voice_mask_to_text(
                        text_enc_time_expanded, voiced_mask)

            # whether to use the unvoiced bias in the attribute predictor
            # circumvent in-place modification
            f0_target = f0.clone()
            if self.ap_use_unvoiced_bias:
                f0_target = torch.detach(f0_target * voiced_mask + f0_bias)
            else:
                f0_target = torch.detach(f0_target)

            # fit to log f0 in f0 predictor
            f0_target[voiced_mask.bool()] = torch.log(
                f0_target[voiced_mask.bool()])
            f0_target = f0_target / 6  # scale to ~ [0, 1] in log space
            energy_avg = energy_avg * 2 - 1  # scale to ~ [-1, 1]

            if self.use_first_order_features:
                df0 = self.get_first_order_features(f0_target, out_lens)
                denergy_avg = self.get_first_order_features(
                    energy_avg, out_lens)

                f0_voiced = torch.cat(
                    (f0_target[:, None],  df0[:, None]), dim=1)
                energy_avg = torch.cat(
                        (energy_avg[:, None], denergy_avg[:, None]), dim=1)

                f0_voiced = f0_voiced * 3  # scale to ~ 1 std
                energy_avg = energy_avg * 3  # scale to ~ 1 std
            else:
                f0_voiced = f0_target * 2  # scale to ~ 1 std
                energy_avg = energy_avg * 1.4  # scale to ~ 1 std

            f0_model_outputs = self.f0_pred_module(
                    text_enc_time_expanded, torch.detach(speaker_vecs),
                    f0_voiced, out_lens)

            energy_model_outputs = self.energy_pred_module(
                    text_enc_time_expanded, torch.detach(speaker_vecs),
                    energy_avg, out_lens)

        outputs = {'z_mel': z_mel,
                   'log_det_W_list': log_det_W_list,
                   'log_s_list': log_s_list,
                   'duration_model_outputs': duration_model_outputs,
                   'f0_model_outputs': f0_model_outputs,
                   'energy_model_outputs': energy_model_outputs,
                   'vpred_model_outputs': vpred_model_outputs,
                   'attn_soft': attn_soft,
                   'attn': attn,
                   'text_embeddings': text_embeddings,
                   'attn_logprob': attn_logprob
                   }

        return outputs

    def infer(self, speaker_id, text, sigma, sigma_dur=0.8, sigma_f0=0.8,
              sigma_energy=0.8, token_dur_scaling=1.0, token_duration_max=100,
              speaker_id_text=None, speaker_id_attributes=None, dur=None,
              f0=None, energy_avg=None, voiced_mask=None, f0_mean=0.0,
              f0_std=0.0, energy_mean=0.0, energy_std=0.0, audio_embedding=None):
        batch_size = text.shape[0]
        n_tokens = text.shape[1]
        if audio_embedding is not None:
            spk_vec = audio_embedding
        else:
            spk_vec = self.encode_speaker(speaker_id)
        spk_vec_text, spk_vec_attributes = spk_vec, spk_vec
        # TODO (Sam): spk_vec_text used in duration, spk_vec_attributes in pitch and "v_pred"
        if speaker_id_text is not None:
            if audio_embedding is not None:
                spk_vec_text = audio_embedding
            else:
                spk_vec_text = self.encode_speaker(speaker_id_text)
        if speaker_id_attributes is not None:
            if audio_embedding is not None:
                spk_vec_attributes = audio_embedding
            else:
                spk_vec_attributes = self.encode_speaker(speaker_id_attributes)

        txt_enc, txt_emb = self.encode_text(text, None)
        if dur is None:
            # get token durations
            z_dur = torch.cuda.FloatTensor(batch_size, 1, n_tokens)
            z_dur = z_dur.normal_() * sigma_dur

            dur = self.dur_pred_layer.infer(z_dur, txt_enc, spk_vec_text)
            if dur.shape[-1] < txt_enc.shape[-1]:
                to_pad = txt_enc.shape[-1] - dur.shape[2]
                pad_fn = nn.ReplicationPad1d((0, to_pad))
                dur = pad_fn(dur)
            dur = dur[:, 0]
            dur = dur.clamp(0, token_duration_max)
            dur = dur * token_dur_scaling if token_dur_scaling > 0 else dur
            dur = (dur + 0.5).floor().int()

        out_lens = dur.sum(1).long().cpu() if dur.shape[0] != 1 else [dur.sum(1)]
        max_n_frames = max(out_lens)

        out_lens = torch.LongTensor(out_lens).to(txt_enc.device)

        txt_enc_time_expanded = self.length_regulator(
            txt_enc.transpose(1, 2), dur).transpose(1, 2)
        if not self.is_attribute_unconditional():
            # if explicitly modeling attributes
            if voiced_mask is None:
                if self.use_vpred_module:
                    # get logits
                    voiced_mask = self.v_pred_module.infer(
                        None, txt_enc_time_expanded, spk_vec_attributes)
                    voiced_mask = (torch.sigmoid(voiced_mask[:, 0]) > 0.5)
                    voiced_mask = voiced_mask.float()

            ap_txt_enc_time_expanded = txt_enc_time_expanded
            # voice mask augmentation only used for attribute prediction
            if self.ap_use_voiced_embeddings:
                ap_txt_enc_time_expanded = self.apply_voice_mask_to_text(
                    txt_enc_time_expanded, voiced_mask)

            f0_bias = 0
            # unvoiced bias forward pass
            if self.use_unvoiced_bias:
                f0_bias = self.unvoiced_bias_module(
                    txt_enc_time_expanded.permute(0, 2, 1))
                f0_bias = -f0_bias[..., 0]
                f0_bias = f0_bias * (~voiced_mask.bool()).float()

            if f0 is None:
                n_f0_feature_channels = 2 if self.use_first_order_features else 1
                z_f0 = torch.cuda.FloatTensor(
                    batch_size, n_f0_feature_channels, max_n_frames).normal_() * sigma_f0
                f0 = self.infer_f0(
                    z_f0, ap_txt_enc_time_expanded, spk_vec_attributes,
                    voiced_mask, out_lens)[:, 0]

            if f0_mean > 0.0:
                vmask_bool = voiced_mask.bool()
                f0_mu, f0_sigma = f0[vmask_bool].mean(), f0[vmask_bool].std()
                f0[vmask_bool] = (f0[vmask_bool] - f0_mu) / f0_sigma
                f0_std = f0_std if f0_std > 0 else f0_sigma
                f0[vmask_bool] = f0[vmask_bool] * f0_std + f0_mean

            if energy_avg is None:
                n_energy_feature_channels = 2 if self.use_first_order_features else 1
                z_energy_avg = torch.cuda.FloatTensor(
                    batch_size, n_energy_feature_channels, max_n_frames).normal_() * sigma_energy
                energy_avg = self.infer_energy(
                    z_energy_avg, ap_txt_enc_time_expanded, spk_vec, out_lens)[:, 0]

            # replication pad, because ungrouping with different group sizes
            # may lead to mismatched lengths
            if energy_avg.shape[1] < out_lens[0]:
                to_pad = out_lens[0] - energy_avg.shape[1]
                pad_fn = nn.ReplicationPad1d((0, to_pad))
                f0 = pad_fn(f0[None])[0]
                energy_avg = pad_fn(energy_avg[None])[0]
            if f0.shape[1] < out_lens[0]:
                to_pad = out_lens[0] - f0.shape[1]
                pad_fn = nn.ReplicationPad1d((0, to_pad))
                f0 = pad_fn(f0[None])[0]

            if self.decoder_use_unvoiced_bias:
                context_w_spkvec = self.preprocess_context(
                    txt_enc_time_expanded, spk_vec, out_lens,
                    f0 * voiced_mask + f0_bias, energy_avg)
            else:
                context_w_spkvec = self.preprocess_context(
                    txt_enc_time_expanded, spk_vec, out_lens, f0*voiced_mask,
                    energy_avg)
        else:
            context_w_spkvec = self.preprocess_context(
                txt_enc_time_expanded, spk_vec, out_lens, None,
                None)

        # NOTE (Sam): comment hacky to get to work on cpu
        # NOTE (Sam): not helpful when available but used.
        # if torch.cuda.is_available():
        residual = torch.cuda.FloatTensor(
            batch_size, 80 * self.n_group_size, max_n_frames // self.n_group_size)
        # else:
        # residual = torch.FloatTensor(
        #     batch_size, 80 * self.n_group_size, max_n_frames // self.n_group_size)
            
        residual = residual.normal_() * sigma

        # map from z sample to data
        exit_steps_stack = self.exit_steps.copy()
        mel = residual[:, len(exit_steps_stack) * self.n_early_size:]
        remaining_residual = residual[:, :len(exit_steps_stack)*self.n_early_size]
        unfolded_seq_lens = out_lens//self.n_group_size
        for i, flow_step in enumerate(reversed(self.flows)):
            curr_step = len(self.flows) - i - 1
            mel = flow_step(mel, context_w_spkvec, inverse=True, seq_lens=unfolded_seq_lens)
            if len(exit_steps_stack) > 0 and curr_step == exit_steps_stack[-1]:
                # concatenate the next chunk of z
                exit_steps_stack.pop()
                residual_to_add = remaining_residual[
                    :, len(exit_steps_stack)*self.n_early_size:]
                remaining_residual = remaining_residual[
                    :, :len(exit_steps_stack)*self.n_early_size]
                mel = torch.cat((residual_to_add, mel), 1)

        if self.n_group_size > 1:
            mel = self.fold(mel)
        if self.do_mel_descaling:
            mel = mel * 2 - 5.5

        return {'mel': mel,
                'dur': dur,
                'f0': f0,
                'energy_avg': energy_avg,
                'voiced_mask': voiced_mask
                }

    def infer_f0(self, residual, txt_enc_time_expanded, spk_vec,
                 voiced_mask=None, lens=None):

        f0 = self.f0_pred_module.infer(
                residual, txt_enc_time_expanded, spk_vec, lens)

        if voiced_mask is not None and len(voiced_mask.shape) == 2:
            voiced_mask = voiced_mask[:, None]

        # constants
        if self.ap_pred_log_f0:
            if self.use_first_order_features:
                f0 = f0[:,0:1,:] / 3
            else:
                f0 = f0 / 2
            f0 = f0 * 6
        else:
            f0 = f0 / 6
            f0 = f0 / 640

        if voiced_mask is None:
            voiced_mask = f0 > 0.0
        else:
            voiced_mask = voiced_mask.bool()

        # due to grouping, f0 might be 1 frame short
        voiced_mask = voiced_mask[:,:,:f0.shape[-1]]
        if self.ap_pred_log_f0:
            # if variable is set, decoder sees linear f0
            # mask = f0 > 0.0 if voiced_mask is None else voiced_mask.bool()
            f0[voiced_mask] = torch.exp(f0[voiced_mask])
        f0[~voiced_mask] = 0.0
        return f0

    def infer_energy(self, residual, txt_enc_time_expanded, spk_vec, lens):
        energy = self.energy_pred_module.infer(
                residual, txt_enc_time_expanded, spk_vec, lens)

        # magic constants
        if self.use_first_order_features:
            energy = energy / 3
        else:
            energy = energy / 1.4
        energy = (energy + 1) / 2
        return energy

    def remove_norms(self):
        """Removes spectral and weightnorms from model. Call before inference
        """
        for name, module in self.named_modules():
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0')
                print("Removed spectral norm from {}".format(name))
            except:
                pass
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0_reverse')
                print("Removed spectral norm from {}".format(name))
            except:
                pass
            try:
                nn.utils.remove_weight_norm(module)
                print("Removed wnorm from {}".format(name))
            except:
                pass

DEFAULT_MODEL_CONFIG = {
 # 'n_speakers': <replace-me>,
 # 'n_speaker_dim': <replace-me>,
 'n_text': 185,
 'n_text_dim': 512,
 'n_flows': 8,
 'n_conv_layers_per_step': 4,
 'n_mel_channels': 80,
 'n_hidden': 1024,
 'mel_encoder_n_hidden': 512,
 'dummy_speaker_embedding': False,
 'n_early_size': 2,
 'n_early_every': 2,
 'n_group_size': 2,
 'affine_model': 'wavenet',
 'include_modules': 'decatndpmvpredapm',
 'scaling_fn': 'tanh',
 'matrix_decomposition': 'LUS',
 'learn_alignments': True,
 'use_speaker_emb_for_alignment': False,
 'attn_straight_through_estimator': True,
 'use_context_lstm': True,
 'context_lstm_norm': 'spectral',
 'context_lstm_w_f0_and_energy': True,
 'text_encoder_lstm_norm': 'spectral',
 'n_f0_dims': 1,
 'n_energy_avg_dims': 1,
 'use_first_order_features': False,
 'unvoiced_bias_activation': 'relu',
 'decoder_use_partial_padding': True,
 'decoder_use_unvoiced_bias': True,
 'ap_pred_log_f0': True,
 'ap_use_unvoiced_bias': False,
 'ap_use_voiced_embeddings': True,
 'dur_model_config': {'hparams': {'arch_hparams': {'kernel_size': 3,
    'n_channels': 256,
    'n_layers': 2,
    'out_dim': 1,
    'p_dropout': 0.25},
   'bottleneck_hparams': {'in_dim': 512,
    'non_linearity': 'relu',
    'norm': 'weightnorm',
    'reduction_factor': 16},
   'n_speaker_dim': 16,
   'take_log_of_input': True},
  'name': 'dap'},
 'energy_model_config': {'hparams': {'arch_hparams': {'kernel_size': 3,
    'n_channels': 256,
    'n_layers': 2,
    'out_dim': 1,
    'p_dropout': 0.25},
   'bottleneck_hparams': {'in_dim': 512,
    'non_linearity': 'relu',
    'norm': 'weightnorm',
    'reduction_factor': 16},
   'n_speaker_dim': 16,
   'take_log_of_input': False,
   'use_transformer': False},
  'name': 'dap'},
 'f0_model_config': {'hparams': {'arch_hparams': {'kernel_size': 11,
    'n_channels': 256,
    'n_layers': 2,
    'out_dim': 1,
    'p_dropout': 0.5},
   'bottleneck_hparams': {'in_dim': 512,
    'non_linearity': 'relu',
    'norm': 'weightnorm',
    'reduction_factor': 16},
   'n_speaker_dim': 16,
   'take_log_of_input': False,
   'use_transformer': False},
  'name': 'dap'},
 'v_model_config': {'name': 'dap',
  'hparams': {'n_speaker_dim': 16,
   'take_log_of_input': False,
   'bottleneck_hparams': {'in_dim': 512,
    'reduction_factor': 16,
    'norm': 'weightnorm',
    'non_linearity': 'relu'},
   'arch_hparams': {'out_dim': 1,
    'n_layers': 2,
    'n_channels': 256,
    'kernel_size': 3,
    'p_dropout': 0.5,
    'lstm_type': '',
    'use_linear': 1}}}}
