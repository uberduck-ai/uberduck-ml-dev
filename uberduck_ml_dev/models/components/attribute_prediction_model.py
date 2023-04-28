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

from uberduck_ml_dev.models.attentions import FFT
from ..common import (
    ConvNorm,
    Invertible1x1Conv,
    AffineTransformationLayer,
    SplineTransformationLayer,
    ConvLSTMLinear,
)
from .transformer import FFTransformer
from .autoregressive_flow import AR_Step, AR_Back_Step


def get_attribute_prediction_model(config):
    name = config["name"]
    hparams = config["hparams"]
    if name == "dap":
        model = DAP(**hparams)
    elif name == "bgap":
        model = BGAP(**hparams)
    elif name == "agap":
        model = AGAP(**hparams)
    else:
        raise Exception("{} model is not supported".format(name))

    return model


class AttributeProcessing:
    def __init__(self, take_log_of_input=False):
        super(AttributeProcessing).__init__()
        self.take_log_of_input = take_log_of_input

    def normalize(self, x):
        if self.take_log_of_input:
            x = torch.log(x + 1)
        return x

    def denormalize(self, x):
        if self.take_log_of_input:
            x = torch.exp(x) - 1
        return x


class BottleneckLayerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        reduction_factor,
        norm="weightnorm",
        non_linearity="relu",
        kernel_size=3,
        use_partial_padding=False,
    ):
        super(BottleneckLayerLayer, self).__init__()

        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            fn = ConvNorm(
                in_dim,
                reduced_dim,
                kernel_size=kernel_size,
                use_weight_norm=(norm == "weightnorm"),
            )
            if norm == "instancenorm":
                fn = nn.Sequential(fn, nn.InstanceNorm1d(reduced_dim, affine=True))

            self.projection_fn = fn
            self.non_linearity = nn.ReLU()
            if non_linearity == "leakyrelu":
                self.non_linearity = nn.LeakyReLU()

    def forward(self, x):
        if self.reduction_factor > 1:
            x = self.projection_fn(x)
            x = self.non_linearity(x)
        return x


class DAP(nn.Module):
    def __init__(
        self,
        n_speaker_dim,
        bottleneck_hparams,
        take_log_of_input,
        arch_hparams,
        use_transformer=False,
    ):
        super(DAP, self).__init__()
        self.attribute_processing = AttributeProcessing(take_log_of_input)
        self.bottleneck_layer = BottleneckLayerLayer(**bottleneck_hparams)

        arch_hparams["in_dim"] = self.bottleneck_layer.out_dim + n_speaker_dim
        if use_transformer:
            self.feat_pred_fn = FFTransformer(**arch_hparams)
        else:
            self.feat_pred_fn = ConvLSTMLinear(**arch_hparams)

    def forward(self, txt_enc, spk_emb, x, lens):
        if x is not None:
            x = self.attribute_processing.normalize(x)

        txt_enc = self.bottleneck_layer(txt_enc)
        spk_emb_expanded = spk_emb[..., None].expand(-1, -1, txt_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)

        x_hat = self.feat_pred_fn(context, lens)

        outputs = {"x_hat": x_hat, "x": x}
        return outputs

    def infer(self, z, txt_enc, spk_emb, lens=None):
        x_hat = self.forward(txt_enc, spk_emb, x=None, lens=lens)["x_hat"]
        x_hat = self.attribute_processing.denormalize(x_hat)
        return x_hat


class BGAP(torch.nn.Module):
    def __init__(
        self,
        n_in_dim,
        n_speaker_dim,
        bottleneck_hparams,
        n_flows,
        n_group_size,
        n_layers,
        with_dilation,
        kernel_size,
        scaling_fn,
        take_log_of_input=False,
        n_channels=1024,
        use_quadratic=False,
        n_bins=8,
        n_spline_steps=2,
    ):
        super(BGAP, self).__init__()
        # assert(n_group_size % 2 == 0)
        self.n_flows = n_flows
        self.n_group_size = n_group_size
        self.transforms = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.n_speaker_dim = n_speaker_dim
        self.scaling_fn = scaling_fn
        self.attribute_processing = AttributeProcessing(take_log_of_input)
        self.n_spline_steps = n_spline_steps
        self.bottleneck_layer = BottleneckLayerLayer(**bottleneck_hparams)
        n_txt_reduced_dim = self.bottleneck_layer.out_dim
        context_dim = n_txt_reduced_dim * n_group_size + n_speaker_dim

        if self.n_group_size > 1:
            self.unfold_params = {
                "kernel_size": (n_group_size, 1),
                "stride": n_group_size,
                "padding": 0,
                "dilation": 1,
            }
            self.unfold = nn.Unfold(**self.unfold_params)

        for k in range(n_flows):
            self.convinv.append(Invertible1x1Conv(n_in_dim * n_group_size))
            if k >= n_flows - self.n_spline_steps:
                left = -3
                right = 3
                top = 3
                bottom = -3
                self.transforms.append(
                    SplineTransformationLayer(
                        n_in_dim * n_group_size,
                        context_dim,
                        n_layers,
                        with_dilation=with_dilation,
                        kernel_size=kernel_size,
                        scaling_fn=scaling_fn,
                        n_channels=n_channels,
                        top=top,
                        bottom=bottom,
                        left=left,
                        right=right,
                        use_quadratic=use_quadratic,
                        n_bins=n_bins,
                    )
                )
            else:
                self.transforms.append(
                    AffineTransformationLayer(
                        n_in_dim * n_group_size,
                        context_dim,
                        n_layers,
                        with_dilation=with_dilation,
                        kernel_size=kernel_size,
                        scaling_fn=scaling_fn,
                        affine_model="simple_conv",
                        n_channels=n_channels,
                    )
                )

    def fold(self, data):
        """Inverse of the self.unfold(data.unsqueeze(-1)) operation used for
        the grouping or "squeeze" operation on input

        Args:
            data: B x C x T tensor of temporal data
        """
        output_size = (data.shape[2] * self.n_group_size, 1)
        data = nn.functional.fold(
            data, output_size=output_size, **self.unfold_params
        ).squeeze(-1)
        return data

    def preprocess_context(self, txt_emb, speaker_vecs, std_scale=None):
        if self.n_group_size > 1:
            txt_emb = self.unfold(txt_emb[..., None])
        speaker_vecs = speaker_vecs[..., None].expand(-1, -1, txt_emb.shape[2])
        context = torch.cat((txt_emb, speaker_vecs), 1)
        return context

    def forward(self, txt_enc, spk_emb, x, lens):
        """x<tensor>: duration or pitch or energy average"""
        assert txt_enc.size(2) >= x.size(1)
        if len(x.shape) == 2:
            # add channel dimension
            x = x[:, None]
        txt_enc = self.bottleneck_layer(txt_enc)

        # lens including padded values
        lens_grouped = (lens // self.n_group_size).long()
        context = self.preprocess_context(txt_enc, spk_emb)
        x = self.unfold(x[..., None])
        log_s_list, log_det_W_list = [], []
        for k in range(self.n_flows):
            x, log_s = self.transforms[k](x, context, seq_lens=lens_grouped)
            x, log_det_W = self.convinv[k](x)
            log_det_W_list.append(log_det_W)
            log_s_list.append(log_s)
        # prepare outputs
        outputs = {"z": x, "log_det_W_list": log_det_W_list, "log_s_list": log_s_list}

        return outputs

    def infer(self, z, txt_enc, spk_emb, seq_lens):
        txt_enc = self.bottleneck_layer(txt_enc)
        context = self.preprocess_context(txt_enc, spk_emb)
        lens_grouped = (seq_lens // self.n_group_size).long()
        z = self.unfold(z[..., None])
        for k in reversed(range(self.n_flows)):
            z = self.convinv[k](z, inverse=True)
            z = self.transforms[k].forward(
                z, context, inverse=True, seq_lens=lens_grouped
            )
        # z mapped to input domain
        x_hat = self.fold(z)
        # pad on the way out
        return x_hat


class AGAP(torch.nn.Module):
    def __init__(
        self,
        n_in_dim,
        n_speaker_dim,
        n_flows,
        n_hidden,
        n_lstm_layers,
        bottleneck_hparams,
        scaling_fn="exp",
        take_log_of_input=False,
        p_dropout=0.0,
        setup="",
        spline_flow_params=None,
        n_group_size=1,
    ):
        super(AGAP, self).__init__()
        self.flows = torch.nn.ModuleList()
        self.n_group_size = n_group_size
        self.n_speaker_dim = n_speaker_dim
        self.attribute_processing = AttributeProcessing(take_log_of_input)
        self.n_in_dim = n_in_dim
        self.bottleneck_layer = BottleneckLayerLayer(**bottleneck_hparams)
        n_txt_reduced_dim = self.bottleneck_layer.out_dim

        if self.n_group_size > 1:
            self.unfold_params = {
                "kernel_size": (n_group_size, 1),
                "stride": n_group_size,
                "padding": 0,
                "dilation": 1,
            }
            self.unfold = nn.Unfold(**self.unfold_params)

        if spline_flow_params is not None:
            spline_flow_params["n_in_channels"] *= self.n_group_size

        for i in range(n_flows):
            if i % 2 == 0:
                self.flows.append(
                    AR_Step(
                        n_in_dim * n_group_size,
                        n_speaker_dim,
                        n_txt_reduced_dim * n_group_size,
                        n_hidden,
                        n_lstm_layers,
                        scaling_fn,
                        spline_flow_params,
                    )
                )
            else:
                self.flows.append(
                    AR_Back_Step(
                        n_in_dim * n_group_size,
                        n_speaker_dim,
                        n_txt_reduced_dim * n_group_size,
                        n_hidden,
                        n_lstm_layers,
                        scaling_fn,
                        spline_flow_params,
                    )
                )

    def fold(self, data):
        """Inverse of the self.unfold(data.unsqueeze(-1)) operation used for
        the grouping or "squeeze" operation on input

        Args:
            data: B x C x T tensor of temporal data
        """
        output_size = (data.shape[2] * self.n_group_size, 1)
        data = nn.functional.fold(
            data, output_size=output_size, **self.unfold_params
        ).squeeze(-1)
        return data

    def preprocess_context(self, txt_emb, speaker_vecs):
        if self.n_group_size > 1:
            txt_emb = self.unfold(txt_emb[..., None])
        speaker_vecs = speaker_vecs[..., None].expand(-1, -1, txt_emb.shape[2])
        context = torch.cat((txt_emb, speaker_vecs), 1)
        return context

    def forward(self, txt_emb, spk_emb, x, lens):
        """x<tensor>: duration or pitch or energy average"""

        x = x[:, None] if len(x.shape) == 2 else x  # add channel dimension
        if self.n_group_size > 1:
            x = self.unfold(x[..., None])
        x = x.permute(2, 0, 1)  # permute to time, batch, dims
        x = self.attribute_processing.normalize(x)

        txt_emb = self.bottleneck_layer(txt_emb)
        context = self.preprocess_context(txt_emb, spk_emb)
        context = context.permute(2, 0, 1)  # permute to time, batch, dims

        lens_groupped = (lens / self.n_group_size).long()
        log_s_list = []
        for i, flow in enumerate(self.flows):
            x, log_s = flow(x, context, lens_groupped)
            log_s_list.append(log_s)

        x = x.permute(1, 2, 0)  # x mapped to z
        log_s_list = [log_s_elt.permute(1, 2, 0) for log_s_elt in log_s_list]
        outputs = {"z": x, "log_s_list": log_s_list, "log_det_W_list": []}
        return outputs

    def infer(self, z, txt_emb, spk_emb, seq_lens=None):
        if self.n_group_size > 1:
            n_frames = z.shape[2]
            z = self.unfold(z[..., None])
        z = z.permute(2, 0, 1)  # permute to time, batch, dims

        txt_emb = self.bottleneck_layer(txt_emb)
        context = self.preprocess_context(txt_emb, spk_emb)
        context = context.permute(2, 0, 1)  # permute to time, batch, dims

        for i, flow in enumerate(reversed(self.flows)):
            z = flow.infer(z, context)

        x_hat = z.permute(1, 2, 0)
        if self.n_group_size > 1:
            x_hat = self.fold(x_hat)
            if n_frames > x_hat.shape[2]:
                m = nn.ReflectionPad1d((0, n_frames - x_hat.shape[2]))
                x_hat = m(x_hat)

        x_hat = self.attribute_processing.denormalize(x_hat)
        return x_hat


class F0Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        spk_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if spk_emb is not None:
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x
