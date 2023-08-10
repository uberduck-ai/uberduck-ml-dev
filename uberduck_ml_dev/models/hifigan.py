# NOTE (Sam): this was used for radtts training inference and inference on uberduck
# the main differences is in how parameters are passed
__all__ = [
    "HiFiGanGenerator",
    "ResBlock1",
    "ResBlock2",
    "Generator",
    "DiscriminatorP",
    "MultiPeriodDiscriminator",
    "DiscriminatorS",
    "MultiScaleDiscriminator",
    "feature_loss",
    "discriminator_loss",
    "generator_loss",
    "LRELU_SLOPE",
    "AttrDict",
    "build_env",
    "init_weights",
    "apply_weight_norm",
    "get_padding",
]

import json
import numpy as np

import os
import shutil
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from .common import ResBlock1, ResBlock2
from .rvc.rvc import (
    SourceModuleHnNSF,
)  # TODO (Sam): we should switch the direction of this import
from .utils import load_pretrained, filter_valid_args

# NOTE(zach): This is config_v1 from https://github.com/jik876/hifi-gan.
# TODO (Sam): try the config from https://dl.fbaipublicfiles.com/voicebox/paper.pdf
DEFAULTS = {
    "resblock": "1",
    "upsample_rates": [8, 8, 2, 2],  # RVC is 10,10,2,2
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "p_blur": 0.0,
}


def _load_uninitialized(device="cpu", config_overrides=None):
    dev = torch.device(device)
    config_dict = DEFAULTS

    if config_overrides is not None:
        config_overrides = filter_valid_args(Generator.__init__, **config_overrides)
        config_dict.update(config_overrides)
    generator = Generator(**config_dict).to(dev)
    return generator


# NOTE (Sam): this is the loading method used by radtts
# TODO (Sam): combine loading methods
# def get_vocoder(hifi_gan_config_path, hifi_gan_checkpoint_path):
#     print("Getting vocoder")

#     with open(hifi_gan_config_path) as f:
#         hifigan_config = json.load(f)


#     h = AttrDict(hifigan_config)
#     hifigan_config["p_blur"] = 0.0
#     model = Generator(**h)
#     print("Loading pretrained model...")
#     load_pretrained(model, hifi_gan_checkpoint_path)
#     print("Got pretrained model...")
#     model.eval()
#     return model
def get_vocoder(hifi_gan_config_path, hifi_gan_checkpoint_path):
    print("Getting vocoder")

    with open(hifi_gan_config_path) as f:
        config_overrides = json.load(f)
    model = _load_uninitialized(config_overrides=config_overrides)
    state_dict = torch.load(hifi_gan_checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["generator"])
    model.eval()
    return model


LRELU_SLOPE = 0.1


class Generator(torch.nn.Module):
    __constants__ = ["lrelu_slope", "num_kernels", "num_upsamples", "p_blur"]

    def __init__(
        self,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        p_blur,
        weight_norm_conv=True,
        initial_channel=80,
        conv_post_bias=True,
        use_noise_convs=False,
        sr=22050,
        is_half=False,
        gin_channels=0,
        harmonic_num=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_post_bias = conv_post_bias
        self.use_noise_convs = use_noise_convs
        # TODO (Sam): detect this automatically
        if weight_norm_conv:
            self.conv_pre = weight_norm(
                Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
            )
        else:
            self.conv_pre = Conv1d(
                initial_channel, upsample_initial_channel, 7, 1, padding=3
            )
        if use_noise_convs:
            self.noise_convs = nn.ModuleList()
            self.m_source = SourceModuleHnNSF(
                sampling_rate=sr, harmonic_num=harmonic_num, is_half=is_half
            )
            self.upp = np.prod(upsample_rates)

        self.p_blur = p_blur
        self.gaussian_blur_fn = None
        if self.p_blur > 0.0:
            # self.gaussian_blur_fn = GaussianBlurAugmentation(
            #     h.gaussian_blur["kernel_size"], h.gaussian_blur["sigmas"], self.p_blur
            # )
            raise Exception(
                "Gaussian blur is not supported in this version of the code."
            )

        else:
            self.gaussian_blur_fn = nn.Identity()
        self.lrelu_slope = LRELU_SLOPE

        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if use_noise_convs:
                c_cur = upsample_initial_channel // (2 ** (i + 1))
                if i + 1 < len(upsample_rates):
                    stride_f0 = np.prod(upsample_rates[i + 1 :])
                    self.noise_convs.append(
                        Conv1d(
                            1,
                            c_cur,
                            kernel_size=stride_f0 * 2,
                            stride=stride_f0,
                            padding=stride_f0 // 2,
                        )
                    )
                else:
                    self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resblock_list = nn.ModuleList()
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                resblock_list.append(resblock(ch, k, d))
            self.resblocks.append(resblock_list)

        if weight_norm_conv:
            self.conv_post = weight_norm(
                Conv1d(ch, 1, 7, 1, padding=3, bias=conv_post_bias)
            )
        else:
            self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=conv_post_bias)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def load_state_dict(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if "resblocks" in k:
                parts = k.split(".")
                # only do this is the checkpoint type is older
                if len(parts) == 5:
                    layer = int(parts[1])
                    new_layer = f"{layer//3}.{layer%3}"
                    new_k = f"resblocks.{new_layer}.{'.'.join(parts[2:])}"
            new_state_dict[new_k] = v
        super().load_state_dict(new_state_dict)

    def forward(self, x, f0=None, g=None):
        if self.use_noise_convs:
            har_source, noi_source, uv = self.m_source(f0, self.upp)
            har_source = har_source.transpose(1, 2)
        if self.p_blur > 0.0:
            x = self.gaussian_blur_fn(x)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i, (upsample_layer, resblock_group) in enumerate(
            zip(self.ups, self.resblocks)
        ):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
            if self.use_noise_convs:
                x += self.noise_convs[i](har_source)
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblock_group:
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for group in self.resblocks:
            for block in group:
                block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
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
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# NOTE (Sam): quite different from rvc parameters here
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]
        scale_discs = [
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
            DiscriminatorS(),
            DiscriminatorS(),
        ]

        period_discs = [
            DiscriminatorP(i, use_spectral_norm=False) for i in periods
        ]  # False is hifigan setting, parameterizable in rvc

        self.mpd = nn.ModuleList(period_discs)
        self.msd = nn.ModuleList(scale_discs)
        self.scale_meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.msd + self.mpd):
            if i in [1, 2]:  # hacky, matches OG hifigan
                y = self.scale_meanpools[i - 1](y)
                y_hat = self.scale_meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# TODO (Sam): this should be moved into Generator
class GeneratorVITS(torch.nn.Module):
    __constants__ = ["lrelu_slope", "num_kernels", "num_upsamples", "p_blur"]

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        gaussian_blur=dict(p_blurring=0),
        weight_norm_pre_and_post=False,
        use_f0=False,
        use_nsf=False,
        use_noise_convs=False,
        use_conv_post_bias=False,
        sampling_rate=None,
    ):
        super().__init__()
        self.use_f0 = use_f0
        self.use_nsf = use_nsf
        if self.use_f0:
            self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        if self.use_nsf:
            assert sampling_rate is not None
            self.m_source = SourceModuleHnNSFVITS(sampling_rate, harmonic_num=8)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        if weight_norm_pre_and_post:
            conv_pre = weight_norm(conv_pre)
        self.conv_pre = conv_pre
        self.p_blur = gaussian_blur["p_blurring"]
        self.gaussian_blur_fn = None
        if self.p_blur > 0.0:
            raise Exception(
                "Gaussian blur is not supported in this version of the code."
            )
            # self.gaussian_blur_fn = GaussianBlurAugmentation(h.gaussian_blur['kernel_size'], h.gaussian_blur['sigmas'], self.p_blur)
        else:
            self.gaussian_blur_fn = nn.Identity()
        self.lrelu_slope = LRELU_SLOPE

        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        if use_noise_convs:
            self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_channels = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        out_channels,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if use_noise_convs:
                if i + 1 < self.num_upsamples:
                    stride_f0 = np.prod(upsample_rates[i + 1 :])
                    self.noise_convs.append(
                        Conv1d(
                            1,
                            out_channels,
                            kernel_size=stride_f0 * 2,
                            stride=stride_f0,
                            padding=stride_f0 // 2,
                        )
                    )
                else:
                    self.noise_convs.append(Conv1d(1, out_channels, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=use_conv_post_bias)
        if weight_norm_pre_and_post:
            conv_post = weight_norm(conv_post)
        self.conv_post = conv_post
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None, f0=None):
        # from einops import rearrange

        if f0 is not None:
            assert self.use_f0
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            signal_source, noise_source, uv = self.m_source(f0)
            signal_source = signal_source.transpose(1, 2)
        if self.p_blur > 0.0:
            raise Exception(
                "Gaussian blur is not supported in this version of the code."
            )
            # x = self.gaussian_blur_fn(x)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for idx, upsample_layer in enumerate(self.ups):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
            if f0 is not None:
                x_source = self.noise_convs[idx](signal_source)
                x = x + x_source
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for j in range(self.num_kernels):
                resblock = self.resblocks[idx * self.num_kernels + j]
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for group in self.resblocks:
            for block in group:
                block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def padDiff(x):
    return F.pad(
        F.pad(x, (0, 0, -1, 1), "constant", 0) - x, (0, 0, 0, -1), "constant", 0
    )


class SineGenVITS(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGenVITS, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            fn = torch.multiply(
                f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
            )

            # generate sine waveforms
            sine_waves = self._f02sine(fn) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSFVITS(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super(SourceModuleHnNSFVITS, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGenVITS(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
