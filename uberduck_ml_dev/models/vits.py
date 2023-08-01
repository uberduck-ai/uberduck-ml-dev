import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

from . import common
from .components.attribute_prediction_model import (
    get_attribute_prediction_model,
    F0Decoder,
)
from .components.encoders.duration import (
    StochasticDurationPredictor,
    DurationPredictor,
)
from .components.encoders.resnet_speaker_encoder import (
    get_pretrained_model,
)
from ..utils.utils import (
    get_padding,
    rand_slice_segments,
    sequence_mask,
    generate_path,
)

from .hifigan import Generator
from .. import monotonic_align
from .components.attentions import VITSEncoder

F0_MODEL_CONFIG = {
    "name": "dap",
    "hparams": {
        # "n_speaker_dim": 16,
        "n_speaker_dim": 512,
        "bottleneck_hparams": {
            # "in_dim": 512,
            "in_dim": 192,
            "reduction_factor": 16,
            "norm": "weightnorm",
            "non_linearity": "relu",
        },
        "take_log_of_input": False,
        "use_transformer": False,
        "arch_hparams": {
            "out_dim": 1,
            "n_layers": 2,
            "n_channels": 256,
            "kernel_size": 11,
            "p_dropout": 0.5,
        },
    },
}
ENERGY_MODEL_CONFIG = {
    "name": "dap",
    "hparams": {
        # "n_speaker_dim": 16,
        "n_speaker_dim": 512,
        "bottleneck_hparams": {
            # "in_dim": 512,
            "in_dim": 192,
            "reduction_factor": 16,
            "norm": "weightnorm",
            "non_linearity": "relu",
        },
        "take_log_of_input": False,
        "use_transformer": False,
        "arch_hparams": {
            "out_dim": 1,
            "n_layers": 2,
            "n_channels": 256,
            "kernel_size": 3,
            "p_dropout": 0.25,
        },
    },
}

SOVITS_MODEL_CONFIG = {
    "inter_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    # NOTE(zach): default sovits config hs [8, 8, 2, 2, 2]
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    # NOTE(zach): default sovits config has [16, 16, 4, 4, 4]
    "upsample_kernel_sizes": [16, 16, 4, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": False,
    "ssl_dim": 256,
}


class SovitsEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = VITSEncoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

    def forward(self, x, x_mask, f0=None, noise_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale) * x_mask

        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = VITSEncoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        local_conditioning_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.local_conditioning_channels = local_conditioning_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                common.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    local_conditioning_channels=local_conditioning_channels,
                )
            )
            self.flows.append(common.Flip())

    def forward(self, x, x_mask, g=None, reverse=False, local_conditioning=None):
        if reverse:
            for flow in reversed(self.flows):
                x = flow(
                    x,
                    x_mask,
                    g=g,
                    reverse=reverse,
                    local_conditioning=local_conditioning,
                )
        else:
            for flow in self.flows:
                x, _ = flow(
                    x,
                    x_mask,
                    g=g,
                    reverse=reverse,
                    local_conditioning=local_conditioning,
                )
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        local_conditioning_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = common.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
            local_conditioning_channels=local_conditioning_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, local_conditioning=None):
        # x has shape [b, c, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g, local_conditioning=local_conditioning)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


# class Generator(torch.nn.Module):
#     def __init__(
#         self,
#         initial_channel,
#         resblock,
#         resblock_kernel_sizes,
#         resblock_dilation_sizes,
#         upsample_rates,
#         upsample_initial_channel,
#         upsample_kernel_sizes,
#         gin_channels=0,
#     ):
#         super(Generator, self).__init__()
#         self.num_kernels = len(resblock_kernel_sizes)
#         self.num_upsamples = len(upsample_rates)
#         self.conv_pre = Conv1d(
#             initial_channel, upsample_initial_channel, 7, 1, padding=3
#         )
#         resblock = common.ResBlock1 if resblock == "1" else common.ResBlock2
#
#         self.ups = nn.ModuleList()
#         for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
#             self.ups.append(
#                 weight_norm(
#                     ConvTranspose1d(
#                         upsample_initial_channel // (2**i),
#                         upsample_initial_channel // (2 ** (i + 1)),
#                         k,
#                         u,
#                         padding=(k - u) // 2,
#                     )
#                 )
#             )
#
#         self.resblocks = nn.ModuleList()
#         for i in range(len(self.ups)):
#             ch = upsample_initial_channel // (2 ** (i + 1))
#             for j, (k, d) in enumerate(
#                 zip(resblock_kernel_sizes, resblock_dilation_sizes)
#             ):
#                 self.resblocks.append(resblock(ch, k, d))
#
#         self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
#         self.ups.apply(init_weights)
#
#         if gin_channels != 0:
#             self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
#
#     def forward(self, x, g=None):
#         x = self.conv_pre(x)
#         if g is not None:
#             x = x + self.cond(g)
#
#         for i in range(self.num_upsamples):
#             x = F.leaky_relu(x, common.LRELU_SLOPE)
#             x = self.ups[i](x)
#             xs = None
#             for j in range(self.num_kernels):
#                 if xs is None:
#                     xs = self.resblocks[i * self.num_kernels + j](x)
#                 else:
#                     xs += self.resblocks[i * self.num_kernels + j](x)
#             x = xs / self.num_kernels
#         x = F.leaky_relu(x)
#         x = self.conv_post(x)
#         x = torch.tanh(x)
#
#         return x
#
#     def remove_weight_norm(self):
#         print("Removing weight norm...")
#         for l in self.ups:
#             remove_weight_norm(l)
#         for l in self.resblocks:
#             l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, common.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, common.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        use_audio_embedding=False,
        gin_channels=0,
        use_sdp=True,
        use_f0=False,
        use_energy=False,
        use_hubert=False,
        ssl_dim=0,
        decoder_weight_norm=False,
        decoder_use_nsf=False,
        decoder_use_f0=False,
        decoder_use_noise_convs=False,
        decoder_use_conv_post_bias=False,
        sampling_rate=None,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        # self.n_speakers = n_speakers
        self.use_speaker_embedding = n_speakers > 0
        self.use_audio_embedding = use_audio_embedding
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp
        self.use_f0 = use_f0
        self.use_energy = use_energy

        if ssl_dim > 0:
            assert use_hubert
            self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        encoder_args = (
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        # NOTE(zach): n_vocab == 0 means there is no text input, i.e. in voice conversion.
        if n_vocab > 0:
            self.enc_p = TextEncoder(n_vocab, *encoder_args)
            if use_sdp:
                self.dp = StochasticDurationPredictor(
                    hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
                )
            else:
                self.dp = DurationPredictor(
                    hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
                )
        else:
            self.enc_p = SovitsEncoder(*encoder_args)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            weight_norm_pre_and_post=decoder_weight_norm,
            use_nsf=decoder_use_nsf,
            use_f0=decoder_use_f0,
            sampling_rate=sampling_rate,
            use_noise_convs=decoder_use_noise_convs,
            use_conv_post_bias=decoder_use_conv_post_bias,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
            # NOTE(zach): pitch conditioning. Try using 1 since it's just a single f0 value?
            # local_conditioning_channels=1 if self.use_f0 else 0,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
            # local_conditioning_channels=1 if self.use_f0 else 0,
        )

        if self.use_f0:
            # NOTE(zach): us this attribute predictor to match radtts's
            # self.f0_decoder = get_attribute_prediction_model(F0_MODEL_CONFIG)
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels,
            )
            self.emb_uv = nn.Embedding(2, hidden_channels)
        if self.use_energy:
            self.energy_predictor = get_attribute_prediction_model(ENERGY_MODEL_CONFIG)

        # if n_speakers > 1:
        #     self.emb_g = nn.Embedding(n_speakers, gin_channels)
        if self.use_speaker_embedding and self.use_audio_embedding:
            raise ValueError("Cannot use both speaker and audio embedding")
        if self.use_audio_embedding:
            self.emb_audio = get_pretrained_model()
            for _param in self.emb_audio.parameters():
                _param.requires_grad = False
        elif self.use_speaker_embedding:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        sid=None,
        audio_embedding=None,
        f0=None,
        voiced_mask=None,
        energy=None,
        hubert_emb=None,
    ):
        if self.use_f0:
            assert f0 is not None
            f0[voiced_mask.bool()] = torch.log(f0[voiced_mask.bool()])
            f0 = f0 / 6
        if self.use_energy:
            assert energy is not None
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.use_audio_embedding:
            g = audio_embedding.unsqueeze(-1)
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(
            y, y_lengths, g=g, local_conditioning=f0.unsqueeze(1)
        )
        z_p = self.flow(z, y_mask, g=g, local_conditioning=f0.unsqueeze(1))

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(
                -1
            )
            neg_cent2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            neg_cent3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(3)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # F0
        if self.use_f0:
            x_expanded = torch.bmm(x, attn.squeeze(1))  # .transpose(1, 2))
            # NOTE(zach): Taken from RadTTS. scale to ~[0, 1] in log space.
            f0_model_outputs = self.f0_decoder(x_expanded, g.squeeze(2), f0, y_lengths)
        # Energy
        if self.use_energy:
            energy_model_outputs = self.energy_predictor(
                x_expanded, g.squeeze(2), y_lengths
            )

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            {
                "f0_model_outputs": f0_model_outputs if self.use_f0 else None,
                "energy_model_outputs": energy_model_outputs
                if self.use_energy
                else None,
            },
        )

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        max_len=None,
        durations=None,
        zero_shot_input=None,
        audio_embedding=None,
        f0=None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        # if self.n_speakers > 0:
        #     g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        if self.use_audio_embedding and audio_embedding is not None:
            g = audio_embedding.unsqueeze(-1)
        elif self.use_audio_embedding and zero_shot_input is not None:
            g = self.emb_audio(zero_shot_input).unsqueeze(-1)
        else:
            g = None

        if durations is not None:
            w_ceil = durations
        elif self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
        else:
            logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        if f0 is None and self.use_f0:
            assert f0 is None
            x_expanded = torch.bmm(x, attn.squeeze(1).transpose(1, 2))
            f0 = self.f0_decoder.infer(
                None,
                x_expanded,
                g.squeeze(2),
                y_lengths,
            )

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True, local_conditioning=f0)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer_textless(self, c, f0, uv, g=None, noise_scale=0.35, predict_f0=False):
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        g = self.emb_g(g).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        if predict_f0:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
            norm_lf0 = normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        z_p, m_p, logs_p, c_mask = self.enc_p(
            x, x_mask, f0=f0_to_coarse(f0), noise_scale=noise_scale
        )
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        # assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        assert self.use_audio_embedding, "Voice conversion requires audio embedding"
        raise Exception("this code is broken")
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)


def normalize_f0(f0, x_mask, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask


def f0_to_coarse(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse
