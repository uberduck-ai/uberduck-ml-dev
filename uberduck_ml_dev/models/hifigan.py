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


""" from https://github.com/jik876/hifi-gan """

import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# NOTE(zach): This is config_v1 from https://github.com/jik876/hifi-gan.
DEFAULTS = {
    "resblock": "1",
    "upsample_rates": [8, 8, 2, 2],
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
        config_dict.update(config_overrides)
    generator = Generator(**config_dict).to(dev)
    return generator


# NOTE (Sam): denoiser not used here in contrast with radtts repo
# def load_vocoder(vocoder_state_dict, vocoder_config, device="cuda"):
#     h = AttrDict(vocoder_config)
#     if "gaussian_blur" in vocoder_config:
#         vocoder_config["gaussian_blur"]["p_blurring"] = 0.0
#     else:
#         vocoder_config["gaussian_blur"] = {"p_blurring": 0.0}
#         h["gaussian_blur"] = {"p_blurring": 0.0}

#     vocoder = Generator(h)
#     vocoder.load_state_dict(vocoder_state_dict)
#     vocoder.to(device)

#     vocoder.eval()

#     return vocoder

from .utils import load_pretrained


# NOTE (Sam): this is the loading method used by radtts
# TODO (Sam): combine loading methods
def get_vocoder(hifi_gan_config_path, hifi_gan_checkpoint_path):
    print("Getting vocoder")

    with open(hifi_gan_config_path) as f:
        hifigan_config = json.load(f)

    h = AttrDict(hifigan_config)
    hifigan_config["p_blur"] = 0.0
    model = Generator(**h)
    print("Loading pretrained model...")
    load_pretrained(model, hifi_gan_checkpoint_path)
    print("Got pretrained model...")
    model.eval()
    return model


LRELU_SLOPE = 0.1


from uberduck_ml_dev.models.common import ResBlock1, ResBlock2


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
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # TODO (Sam): detect this automatically
        if weight_norm_conv:
            self.conv_pre = weight_norm(
                Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
            )
        else:
            self.conv_pre = Conv1d(
                initial_channel, upsample_initial_channel, 7, 1, padding=3
            )
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

    def forward(self, x):
        if self.p_blur > 0.0:
            x = self.gaussian_blur_fn(x)
        x = self.conv_pre(x)
        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
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


import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


from torch.nn.utils import weight_norm


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
