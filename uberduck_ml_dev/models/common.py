__all__ = [
    "Conv1d",
    "LinearNorm",
    "LocationLayer",
    "Attention",
    "STFT",
    "MelSTFT",
    "ReferenceEncoder",
    "MultiHeadAttention",
    "STL",
    "GST",
    "LayerNorm",
    "Flip",
    "Log",
    "ElementwiseAffine",
    "DDSConv",
    "ConvFlow",
    "WN",
    "ResidualCouplingLayer",
    "ResBlock1",
    "ResBlock2",
    "LRELU_SLOPE",
]

import math

import numpy as np
from numpy import finfo
from scipy.signal import get_window
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.nn import init
from torch.cuda import amp
from librosa.filters import mel as librosa_mel
from librosa.util import pad_center, tiny
from typing import Tuple

from ..utils.utils import *
from .transforms import piecewise_rational_quadratic_transform
from .components.partialconv1d import PartialConv1d as pconv1d

class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        return self.conv(signal)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = Conv1d(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MEL_CHANNELS = 80
SAMPLING_RATE = 22050


# NOTE (Sam): STFTs should get their own file in common folder
class STFT:
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self,
        filter_length=FILTER_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        padding=None,
        device="cpu",
        rank=None,
    ):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        self.padding = padding or (filter_length // 2)

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        if device == "cuda":
            dev = torch.device(f"cuda:{rank}")
            forward_basis = torch.cuda.FloatTensor(
                fourier_basis[:, None, :], device=dev
            )
            inverse_basis = torch.cuda.FloatTensor(
                np.linalg.pinv(scale * fourier_basis).T[:, None, :].astype(np.float32),
                device=dev,
            )
        else:
            forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
            inverse_basis = torch.FloatTensor(
                np.linalg.pinv(scale * fourier_basis).T[:, None, :].astype(np.float32)
            )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            if device == "cuda":
                fft_window = fft_window.cuda(rank)

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
            self.fft_window = fft_window

        self.forward_basis = forward_basis.float()
        self.inverse_basis = inverse_basis.float()

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, -1)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (
                self.padding,
                self.padding,
                0,
                0,
            ),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_basis = self.forward_basis.to(input_data.device)
        forward_transform = F.conv1d(
            input_data,
            Variable(forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        cutoff = self.filter_length // 2 + 1
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)],
            dim=1,
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


MEL_FMIN = 0.0
MEL_FMAX = 8000.0


class MelSTFT:
    def __init__(
        self,
        filter_length=FILTER_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        sampling_rate=SAMPLING_RATE,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
        device="cpu",
        padding=None,
        rank=None,
    ):
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        if padding is None:
            padding = filter_length // 2

        self.stft_fn = STFT(
            filter_length,
            hop_length,
            win_length,
            device=device,
            rank=rank,
            padding=padding,
        )
        mel_basis = librosa_mel(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        if device == "cuda":
            mel_basis = mel_basis.cuda()
        self.mel_basis = mel_basis

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def spec_to_mel(self, spec):
        basis = self.mel_basis.to(spec.device)
        mel_output = torch.matmul(basis, spec)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def spectrogram(self, y):
        assert y.min() >= -1
        assert y.max() <= 1
        magnitudes, phases = self.stft_fn.transform(y)
        return magnitudes.data

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert y.min() >= -1
        assert y.max() <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        return self.spec_to_mel(magnitudes)

    def griffin_lim(self, mel_spectrogram, n_iters=30):
        mel_dec = self.spectral_de_normalize(mel_spectrogram)
        # Float cast required for fp16 training.
        mel_dec = mel_dec.transpose(0, 1).cpu().data.float()
        spec_from_mel = torch.mm(mel_dec, self.mel_basis).transpose(0, 1)
        spec_from_mel *= 1000
        out = griffin_lim(spec_from_mel.unsqueeze(0), self.stft_fn, n_iters=n_iters)
        return out


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, hp):
        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)]
        )

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=hp.ref_enc_filters[-1] * out_channels,
            hidden_size=hp.ref_enc_gru_size,
            batch_first=True,
        )
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths, batch_first=True, enforce_sorted=False
            )

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim, out_features=num_units, bias=False
        )
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False
        )

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(
            torch.split(querys, split_size, dim=2), dim=0
        )  # [h, N, T_q, num_units/h]
        keys = torch.stack(
            torch.split(keys, split_size, dim=2), dim=0
        )  # [h, N, T_k, num_units/h]
        values = torch.stack(
            torch.split(values, split_size, dim=2), dim=0
        )  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim**0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(
            0
        )  # [N, T_q, num_units]

        return out


class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(
            torch.FloatTensor(hp.token_num, hp.token_embedding_size // hp.num_heads)
        )
        d_q = hp.ref_enc_gru_size
        d_k = hp.token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q,
            key_dim=d_k,
            num_units=hp.token_embedding_size,
            num_heads=hp.num_heads,
        )

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = (
            torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        )  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class GST(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed = self.stl(enc_out)

        return style_embed


# Cell
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# Cell
class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


# Cell
class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


# Cell
class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


# Cell
class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        # tail_bound=5.0,
        tail_bound=10.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


# Cell
from ..utils.utils import fused_add_tanh_sigmoid_multiply


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)



class WN_RADTTS(torch.nn.Module):
    """
    Adapted from WN() module in WaveGlow with modififcations to variable names
    """
    def __init__(self, n_in_channels, n_context_dim, n_layers, n_channels,
                 kernel_size=5, affine_activation='softplus',
                 use_partial_padding=True):
        super(WN_RADTTS, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        start = torch.nn.Conv1d(n_in_channels+n_context_dim, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start
        self.softplus = torch.nn.Softplus()
        self.affine_activation = affine_activation
        self.use_partial_padding = use_partial_padding
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = ConvNorm(n_channels, n_channels, kernel_size=kernel_size,
                                dilation=dilation, padding=padding,
                                use_partial_padding=use_partial_padding,
                                use_weight_norm=True)
            # in_layer = nn.Conv1d(n_channels, n_channels, kernel_size,
            #                      dilation=dilation, padding=padding)
            # in_layer = nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)
            res_skip_layer = nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input: Tuple[torch.Tensor, torch.Tensor], seq_lens: torch.Tensor = None):
        z, context = forward_input
        z = torch.cat((z, context), 1)  # append context to z as well
        z = self.start(z)
        output = torch.zeros_like(z)
        mask = None
        if seq_lens is not None:
            mask = get_mask_from_lengths(seq_lens).unsqueeze(1).float()
        non_linearity = torch.relu
        if self.affine_activation == 'softplus':
            non_linearity = self.softplus

        for i in range(self.n_layers):
            z = non_linearity(self.in_layers[i](z, mask))
            res_skip_acts = non_linearity(self.res_skip_layers[i](z))
            output = output + res_skip_acts

        output = self.end(output)  # [B, dim, seq_len]
        return output



# Cell
class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


# Cell


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


LRELU_SLOPE = 0.1


mel_basis, hann_window = {}, {}

def mel_spectrogram_torch(
    y,
    n_fft=FILTER_LENGTH,
    num_mels=N_MEL_CHANNELS,
    sampling_rate=SAMPLING_RATE,
    hop_size=HOP_LENGTH,
    win_size=WIN_LENGTH,
    fmin=MEL_FMIN,
    fmax=MEL_FMAX,
    center=False,
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel(
            sr=sampling_rate, n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    print(y.shape)
    y = torch.nn.functional.pad(
        y,
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    def spectral_normalize(magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    spec = spectral_normalize(spec)

    return spec

def spec_to_mel_torch(spec, n_fft=FILTER_LENGTH, num_mels=N_MEL_CHANNELS, sampling_rate=SAMPLING_RATE, fmin=MEL_FMIN, fmax=MEL_FMAX):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    def spectral_normalize(magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    spec = spectral_normalize(spec)
    return spec

def spectrogram_torch(y, n_fft=FILTER_LENGTH, sampling_rate=SAMPLING_RATE, hop_size=HOP_LENGTH, win_size=WIN_LENGTH, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

# TODO (Sam): unite the get_mel methods
def get_mel(audio, max_wav_value, stft):

    audio_norm = ((max_wav_value - 1) * audio / (np.abs(audio).max())) / max_wav_value
    audio_norm = torch.FloatTensor(audio_norm)
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    # audio_norm = audio / max_wav_value
    # audio_norm = audio_norm.unsqueeze(0)
    # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    # melspec = stft.mel_spectrogram(audio_norm)
    # melspec = torch.squeeze(melspec, 0)
    # if self.do_mel_scaling:
    melspec = (melspec + 5.5) / 2
    # if self.mel_noise_scale > 0:
    #     melspec += torch.randn_like(melspec) * self.mel_noise_scale
    return melspec

class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_text_channels=512,
                 n_att_channels=80, temperature=1.0):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels*2, kernel_size=3,
                     bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels*2, n_att_channels, kernel_size=1,
                     bias=True))

        self.query_proj = nn.Sequential(
            ConvNorm(n_mel_channels, n_mel_channels*2, kernel_size=3,
                     bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels*2, n_mel_channels, kernel_size=1,
                     bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True)
        )

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model
        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None,
                attn_prior=None):
        """Attention mechanism for radtts. Unlike in Flowtron, we have no
        restrictions such as causality etc, since we only need this during
        training.
        Args:
            queries (torch.tensor): B x C x T1 tensor (likely mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                                 (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                                 Final dim T2 should sum to 1
        """
        temp = 0.0005
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries)

        # Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None])**2

        # compute log-likelihood from gaussian
        eps = 1e-8
        attn = -temp * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + eps)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(
                mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob




class AffineTransformationLayer(torch.nn.Module):
    def __init__(self, n_mel_channels, n_context_dim, n_layers,
                 affine_model='simple_conv', with_dilation=True, kernel_size=5,
                 scaling_fn='exp', affine_activation='softplus',
                 n_channels=1024, use_partial_padding=False):
        super(AffineTransformationLayer, self).__init__()
        if affine_model not in ("wavenet", "simple_conv"):
            raise Exception("{} affine model not supported".format(affine_model))
        if isinstance(scaling_fn, list):
            if not all([x in ("translate", "exp", "tanh", "sigmoid") for x in scaling_fn]):
                raise Exception("{} scaling fn not supported".format(scaling_fn))
        else:
            if scaling_fn not in ("translate", "exp", "tanh", "sigmoid"):
                raise Exception("{} scaling fn not supported".format(scaling_fn))

        self.affine_model = affine_model
        self.scaling_fn = scaling_fn
        if affine_model == 'wavenet':
            self.affine_param_predictor = WN_RADTTS(
                int(n_mel_channels/2), n_context_dim, n_layers=n_layers,
                n_channels=n_channels, affine_activation=affine_activation,
                use_partial_padding=use_partial_padding)
        elif affine_model == 'simple_conv':
            self.affine_param_predictor = SimpleConvNet(
                int(n_mel_channels / 2), n_context_dim, n_mel_channels,
                n_layers, with_dilation=with_dilation, kernel_size=kernel_size,
                use_partial_padding=use_partial_padding)
        self.n_mel_channels = n_mel_channels

    def get_scaling_and_logs(self, scale_unconstrained):
        if self.scaling_fn == 'translate':
            s = torch.exp(scale_unconstrained*0)
            log_s = scale_unconstrained*0
        elif self.scaling_fn == 'exp':
            s = torch.exp(scale_unconstrained)
            log_s = scale_unconstrained  # log(exp
        elif self.scaling_fn == 'tanh':
            s = torch.tanh(scale_unconstrained) + 1 + 1e-6
            log_s = torch.log(s)
        elif self.scaling_fn == 'sigmoid':
            s = torch.sigmoid(scale_unconstrained + 10) + 1e-6
            log_s = torch.log(s)
        elif isinstance(self.scaling_fn, list):
            s_list, log_s_list = [], []
            for i in range(scale_unconstrained.shape[1]):
                scaling_i = self.scaling_fn[i]
                if scaling_i == 'translate':
                    s_i = torch.exp(scale_unconstrained[:i]*0)
                    log_s_i = scale_unconstrained[:, i]*0
                elif scaling_i == 'exp':
                    s_i = torch.exp(scale_unconstrained[:, i])
                    log_s_i = scale_unconstrained[:, i]
                elif scaling_i == 'tanh':
                    s_i = torch.tanh(scale_unconstrained[:, i]) + 1 + 1e-6
                    log_s_i = torch.log(s_i)
                elif scaling_i == 'sigmoid':
                    s_i = torch.sigmoid(scale_unconstrained[:, i]) + 1e-6
                    log_s_i = torch.log(s_i)
                s_list.append(s_i[:, None])
                log_s_list.append(log_s_i[:, None])
            s = torch.cat(s_list, dim=1)
            log_s = torch.cat(log_s_list, dim=1)
        return s, log_s

    def forward(self, z, context, inverse=False, seq_lens=None):
        n_half = int(self.n_mel_channels / 2)
        z_0, z_1 = z[:, :n_half], z[:, n_half:]
        if self.affine_model == 'wavenet':
            affine_params = self.affine_param_predictor(
                (z_0, context), seq_lens=seq_lens)
        elif self.affine_model == 'simple_conv':
            z_w_context = torch.cat((z_0, context), 1)
            affine_params = self.affine_param_predictor(
                z_w_context, seq_lens=seq_lens)

        scale_unconstrained = affine_params[:, :n_half, :]
        b = affine_params[:, n_half:, :]
        s, log_s = self.get_scaling_and_logs(scale_unconstrained)

        if inverse:
            z_1 = (z_1 - b) / s
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:
            z_1 = s * z_1 + b
            z = torch.cat((z_0, z_1), dim=1)
            return z, log_s


class ExponentialClass(torch.nn.Module):
    def __init__(self):
        super(ExponentialClass, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dur):
        output = []
        for x_i, dur_i in zip(x, dur):
            expanded = self.expand(x_i, dur_i)
            output.append(expanded)
        output = self.pad(output)
        return output

    def expand(self, x, dur):
        output = []
        for i, frame in enumerate(x):
            expanded_len = int(dur[i] + 0.5)
            expanded = frame.expand(expanded_len, -1)
            output.append(expanded)
        output = torch.cat(output, 0)
        return output

    def pad(self, x):
        output = []
        max_len = max([x[i].size(0) for i in range(len(x))])
        for i, seq in enumerate(x):
            padded = F.pad(
                seq, [0, 0, 0, max_len - seq.size(0)], 'constant', 0.0)
            output.append(padded)
        output = torch.stack(output)
        return output


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear',
                 use_partial_padding=False, use_weight_norm=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_partial_padding = use_partial_padding
        self.use_weight_norm = use_weight_norm
        conv_fn = torch.nn.Conv1d
        if self.use_partial_padding:
            conv_fn = pconv1d
        self.conv = conv_fn(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation,
                            bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if self.use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            conv_signal = self.conv(signal, mask)
        else:
            conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions=3, encoder_embedding_dim=512,
                 encoder_kernel_size=5, norm_fn=nn.BatchNorm1d,
                 lstm_norm_fn=None):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu',
                         use_partial_padding=True),
                norm_fn(encoder_embedding_dim, affine=True))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        if lstm_norm_fn is not None:
            if 'spectral' in lstm_norm_fn:
                print("Applying spectral norm to text encoder LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.spectral_norm
            elif 'weight' in lstm_norm_fn:
                print("Applying weight norm to text encoder LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.weight_norm
            self.lstm = lstm_norm_fn_pntr(self.lstm, 'weight_hh_l0')
            self.lstm = lstm_norm_fn_pntr(self.lstm, 'weight_hh_l0_reverse')

    @amp.autocast(False)
    def forward(self, x, in_lens):
        """
        Args:
            x (torch.tensor): N x C x L padded input of text embeddings
            in_lens (torch.tensor): 1D tensor of sequence lengths
        """
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: improve speed
                curr_x = x[b_ind:b_ind+1, :, :in_lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(F.relu(conv(curr_x)),
                                       0.5, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)

        # recent amp change -- change in_lens to int
        in_lens = in_lens.int().cpu()

        x = nn.utils.rnn.pack_padded_sequence(x, in_lens, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    @amp.autocast(False)
    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c, cache_inverse=False):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))
        self.cache_inverse = cache_inverse

    @amp.autocast(False)
    def forward(self, z, inverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if inverse:
            if not hasattr(self, 'W_inverse'):
                # inverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            if not self.cache_inverse:
                delattr(self, 'W_inverse')
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If inverse=True it does convolution with
    inverse
    """
    def __init__(self, c, cache_inverse=False):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W
        self.cache_inverse = cache_inverse

    def forward(self, z, inverse=False):
        # DO NOT apply n_of_groups, as it doesn't account for padded sequences
        W = self.conv.weight.squeeze()

        if inverse:
            if not hasattr(self, 'W_inverse'):
                # Inverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            if not self.cache_inverse:
                delattr(self, 'W_inverse')
            return z
        else:
            # Forward computation
            log_det_W = torch.logdet(W).clone()
            z = self.conv(z)
            return z, log_det_W


class SimpleConvNet(torch.nn.Module):
    def __init__(self, n_mel_channels, n_context_dim, final_out_channels,
                 n_layers=2, kernel_size=5, with_dilation=True,
                 max_channels=1024, zero_init=True, use_partial_padding=True):
        super(SimpleConvNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_layers = n_layers
        in_channels = n_mel_channels + n_context_dim
        out_channels = -1
        self.use_partial_padding = use_partial_padding
        for i in range(n_layers):
            dilation = 2 ** i if with_dilation else 1
            padding = int((kernel_size*dilation - dilation)/2)
            out_channels = min(max_channels, in_channels * 2)
            self.layers.append(ConvNorm(in_channels, out_channels,
                                        kernel_size=kernel_size, stride=1,
                                        padding=padding, dilation=dilation,
                                        bias=True, w_init_gain='relu',
                                        use_partial_padding=use_partial_padding))
            in_channels = out_channels

        self.last_layer = torch.nn.Conv1d(
            out_channels, final_out_channels, kernel_size=1)

        if zero_init:
            self.last_layer.weight.data *= 0
            self.last_layer.bias.data *= 0

    def forward(self, z_w_context, seq_lens: torch.Tensor = None):
        # seq_lens: tensor array of sequence sequence lengths
        # output should be b x n_mel_channels x z_w_context.shape(2)
        mask = None
        if seq_lens is not None:
            mask = get_mask_from_lengths(seq_lens).unsqueeze(1).float()

        for i in range(self.n_layers):
            z_w_context = self.layers[i](z_w_context, mask)
            z_w_context = torch.relu(z_w_context)

        z_w_context = self.last_layer(z_w_context)
        return z_w_context



# Affine Coupling Layers
class SplineTransformationLayerAR(torch.nn.Module):
    def __init__(self, n_in_channels, n_context_dim, n_layers,
                 affine_model='simple_conv', kernel_size=1, scaling_fn='exp',
                 affine_activation='softplus', n_channels=1024, n_bins=8,
                 left=-6, right=6, bottom=-6, top=6, use_quadratic=False):
        super(SplineTransformationLayerAR, self).__init__()
        self.n_in_channels = n_in_channels  # input dimensions
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.n_bins = n_bins
        self.spline_fn = piecewise_linear_transform
        self.inv_spline_fn = piecewise_linear_inverse_transform
        self.use_quadratic = use_quadratic

        if self.use_quadratic:
            self.spline_fn = unbounded_piecewise_quadratic_transform
            self.inv_spline_fn = unbounded_piecewise_quadratic_transform
            self.n_bins = 2 * self.n_bins + 1
        final_out_channels = self.n_in_channels * self.n_bins

        # autoregressive flow, kernel size 1 and no dilation
        self.param_predictor = SimpleConvNet(
            n_context_dim, 0, final_out_channels, n_layers,
            with_dilation=False, kernel_size=1, zero_init=True,
            use_partial_padding=False)

        # output is unnormalized bin weights

    def normalize(self, z, inverse):
        # normalize to [0, 1]
        if inverse:
            z = (z - self.bottom) / (self.top - self.bottom)
        else:
            z = (z - self.left) / (self.right - self.left)

        return z

    def denormalize(self, z, inverse):
        if inverse:
            z = z * (self.right - self.left) + self.left
        else:
            z = z * (self.top - self.bottom) + self.bottom

        return z

    def forward(self, z, context, inverse=False):
        b_s, c_s, t_s = z.size(0), z.size(1), z.size(2)

        z = self.normalize(z, inverse)

        if z.min() < 0.0 or z.max() > 1.0:
            print('spline z scaled beyond [0, 1]', z.min(), z.max())


        z_reshaped = z.permute(0, 2, 1).reshape(b_s * t_s, -1)
        affine_params = self.param_predictor(context)
        q_tilde = affine_params.permute(0, 2, 1).reshape(b_s * t_s, c_s, -1)
        with amp.autocast(enabled=False):
            if self.use_quadratic:
                w = q_tilde[:, :, :self.n_bins // 2]
                v = q_tilde[:, :, self.n_bins // 2:]
                z_tformed, log_s = self.spline_fn(
                    z_reshaped.float(), w.float(), v.float(), inverse=inverse)
            else:
                z_tformed, log_s = self.spline_fn(
                    z_reshaped.float(), q_tilde.float())

        z = z_tformed.reshape(b_s, t_s, -1).permute(0, 2, 1)
        z = self.denormalize(z, inverse)
        if inverse:
            return z

        log_s = log_s.reshape(b_s, t_s, -1)
        log_s = log_s.permute(0, 2, 1)
        log_s = log_s + c_s * (np.log(self.top - self.bottom) -
                               np.log(self.right - self.left))
        return z, log_s


class DenseLayer(nn.Module):
    def __init__(self, in_dim=1024, sizes=[1024, 1024]):
        super(DenseLayer, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = torch.tanh(linear(x))
        return x

from .components.splines import (piecewise_linear_transform,
                     piecewise_linear_inverse_transform,
                     unbounded_piecewise_quadratic_transform)
                     
class SplineTransformationLayer(torch.nn.Module):
    def __init__(self, n_mel_channels, n_context_dim, n_layers,
                 with_dilation=True, kernel_size=5,
                 scaling_fn='exp', affine_activation='softplus',
                 n_channels=1024, n_bins=8, left=-4, right=4, bottom=-4, top=4,
                 use_quadratic=False):
        super(SplineTransformationLayer, self).__init__()
        self.n_mel_channels = n_mel_channels  # input dimensions
        self.half_mel_channels = int(n_mel_channels/2)  # half, because we split
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.n_bins = n_bins
        self.spline_fn = piecewise_linear_transform
        self.inv_spline_fn = piecewise_linear_inverse_transform
        self.use_quadratic = use_quadratic

        if self.use_quadratic:
            self.spline_fn = unbounded_piecewise_quadratic_transform
            self.inv_spline_fn = unbounded_piecewise_quadratic_transform
            self.n_bins = 2*self.n_bins+1
        final_out_channels = self.half_mel_channels*self.n_bins

        self.param_predictor = SimpleConvNet(
            self.half_mel_channels, n_context_dim, final_out_channels,
            n_layers, with_dilation=with_dilation, kernel_size=kernel_size,
            zero_init=False)

        # output is unnormalized bin weights

    def forward(self, z, context, inverse=False, seq_lens=None):
        b_s, c_s, t_s = z.size(0), z.size(1), z.size(2)

        # condition on z_0, transform z_1
        n_half = self.half_mel_channels
        z_0, z_1 = z[:, :n_half], z[:, n_half:]

        # normalize to [0,1]
        if inverse:
            z_1 = (z_1 - self.bottom)/(self.top - self.bottom)
        else:
            z_1 = (z_1 - self.left)/(self.right - self.left)

        z_w_context = torch.cat((z_0, context), 1)
        affine_params = self.param_predictor(z_w_context, seq_lens)
        z_1_reshaped = z_1.permute(0, 2, 1).reshape(b_s*t_s, -1)
        q_tilde = affine_params.permute(0, 2, 1).reshape(
            b_s*t_s, n_half, self.n_bins)

        with autocast(enabled=False):
            if self.use_quadratic:
                w = q_tilde[:, :, :self.n_bins//2]
                v = q_tilde[:, :, self.n_bins//2:]
                z_1_tformed, log_s = self.spline_fn(
                    z_1_reshaped.float(), w.float(), v.float(),
                    inverse=inverse)
                if not inverse:
                    log_s = torch.sum(log_s, 1)
            else:
                if inverse:
                    z_1_tformed, _dc = self.inv_spline_fn(
                        z_1_reshaped.float(), q_tilde.float(), False)
                else:
                    z_1_tformed, log_s = self.spline_fn(
                        z_1_reshaped.float(), q_tilde.float())

        z_1 = z_1_tformed.reshape(b_s, t_s, -1).permute(0, 2, 1)

        # undo [0, 1] normalization
        if inverse:
            z_1 = z_1 * (self.right - self.left) + self.left
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:  # training
            z_1 = z_1 * (self.top - self.bottom) + self.bottom
            z = torch.cat((z_0, z_1), dim=1)
            log_s = log_s.reshape(b_s, t_s).unsqueeze(1) + \
                n_half*(np.log(self.top - self.bottom) -
                        np.log(self.right-self.left))
            return z, log_s


class ConvLSTMLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=2, n_channels=256,
                 kernel_size=3, p_dropout=0.1, lstm_type='bilstm',
                 use_linear=True):
        super(ConvLSTMLinear, self).__init__()
        self.out_dim = out_dim
        self.lstm_type = lstm_type
        self.use_linear = use_linear
        self.dropout = nn.Dropout(p=p_dropout)

        convolutions = []
        for i in range(n_layers):
            conv_layer = ConvNorm(
                in_dim if i == 0 else n_channels, n_channels,
                kernel_size=kernel_size, stride=1,
                padding=int((kernel_size - 1) / 2), dilation=1,
                w_init_gain='relu')
            conv_layer = torch.nn.utils.weight_norm(
                conv_layer.conv, name='weight')
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        if not self.use_linear:
            n_channels = out_dim

        if self.lstm_type != '':
            use_bilstm = False
            lstm_channels = n_channels
            if self.lstm_type == 'bilstm':
                use_bilstm = True
                lstm_channels = int(n_channels // 2)

            self.bilstm = nn.LSTM(n_channels, lstm_channels, 1,
                                  batch_first=True, bidirectional=use_bilstm)
            lstm_norm_fn_pntr = nn.utils.spectral_norm
            self.bilstm = lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0')
            if self.lstm_type == 'bilstm':
                self.bilstm = lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0_reverse')

        if self.use_linear:
            self.dense = nn.Linear(n_channels, out_dim)

    def run_padded_sequence(self, context, lens):
        context_embedded = []
        for b_ind in range(context.size()[0]):  # TODO: speed up
            curr_context = context[b_ind:b_ind+1, :, :lens[b_ind]].clone()
            for conv in self.convolutions:
                curr_context = self.dropout(F.relu(conv(curr_context)))
            context_embedded.append(curr_context[0].transpose(0, 1))
        context = torch.nn.utils.rnn.pad_sequence(
            context_embedded, batch_first=True)
        return context

    def run_unsorted_inputs(self, fn, context, lens):
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(
            context, lens_sorted, batch_first=True)
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(
            context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens):
        if context.size()[0] > 1:
            context = self.run_padded_sequence(context, lens)
            # to B, D, T
            context = context.transpose(1, 2)
        else:
            for conv in self.convolutions:
                context = self.dropout(F.relu(conv(context)))

        if self.lstm_type != '':
            context = context.transpose(1, 2)
            self.bilstm.flatten_parameters()
            if lens is not None:
                context = self.run_unsorted_inputs(self.bilstm, context, lens)
            else:
                context = self.bilstm(context)[0]
            context = context.transpose(1, 2)

        x_hat = context
        if self.use_linear:
            x_hat = self.dense(context.transpose(1, 2)).transpose(1, 2)

        return x_hat

    def infer(self, z, txt_enc, spk_emb):
        x_hat = self.forward(txt_enc, spk_emb)['x_hat']
        x_hat = self.feature_processing.denormalize(x_hat)
        return x_hat

