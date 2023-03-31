import torch
from torch import nn

from .data.batch import Batch


class Tacotron2Loss(nn.Module):
    def __init__(self, pos_weight):
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight)
        else:
            self.pos_weight = pos_weight

        super().__init__()

    # NOTE (Sam): making function inputs explicit makes less sense in situations like this with obvious subcategories.
    def forward(self, model_output: Batch, target: Batch):
        mel_target, gate_target = target["mel_padded"], target["gate_target"]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_out, mel_out_postnet, gate_out = (
            model_output["mel_outputs"],
            model_output["mel_outputs_postnet"],
            model_output["gate_predicted"],
        )
        mel_loss_batch = nn.MSELoss(reduction="none")(mel_out, mel_target).mean(
            axis=[1, 2]
        ) + nn.MSELoss(reduction="none")(mel_out_postnet, mel_target).mean(axis=[1, 2])

        mel_loss = mel_loss_batch.mean()

        gate_loss_batch = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduce=False
        )(gate_out, gate_target).mean(axis=[1])
        gate_loss = torch.mean(gate_loss_batch)

        return mel_loss, gate_loss, mel_loss_batch, gate_loss_batch


# VITS losses
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
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
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


# RADTTS losses

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
import torch.nn as nn
from torch.nn import functional as F

# from common import get_mask_from_lengths
from .utils.utils import get_mask_from_lengths_radtts as get_mask_from_lengths


def compute_flow_loss(
    z, log_det_W_list, log_s_list, n_elements, n_dims, mask, sigma=1.0
):

    log_det_W_total = 0.0
    for i, log_s in enumerate(log_s_list):
        if i == 0:
            log_s_total = torch.sum(log_s * mask)
            if len(log_det_W_list):
                log_det_W_total = log_det_W_list[i]
        else:
            log_s_total = log_s_total + torch.sum(log_s * mask)
            if len(log_det_W_list):
                log_det_W_total += log_det_W_list[i]

    if len(log_det_W_list):
        log_det_W_total *= n_elements

    z = z * mask
    prior_NLL = torch.sum(z * z) / (2 * sigma * sigma)

    loss = prior_NLL - log_s_total - log_det_W_total

    denom = n_elements * n_dims
    loss = loss / denom
    loss_prior = prior_NLL / denom
    return loss, loss_prior


def compute_regression_loss(x_hat, x, mask, name=False):
    x = x[:, None] if len(x.shape) == 2 else x  # add channel dim
    mask = mask[:, None] if len(mask.shape) == 2 else mask  # add channel dim
    assert len(x.shape) == len(mask.shape)

    x = x * mask
    x_hat = x_hat * mask

    if name == "vpred":
        loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    else:
        loss = F.mse_loss(x_hat, x, reduction="sum")
    loss = loss / mask.sum()

    loss_dict = {"loss_{}".format(name): loss}

    return loss_dict


class AttributePredictionLoss(torch.nn.Module):
    def __init__(self, name, model_config, loss_weight, sigma=1.0):
        super(AttributePredictionLoss, self).__init__()
        self.name = name
        self.sigma = sigma
        self.model_name = model_config["name"]
        self.loss_weight = loss_weight
        self.n_group_size = 1
        if "n_group_size" in model_config["hparams"]:
            self.n_group_size = model_config["hparams"]["n_group_size"]

    def forward(self, model_output, lens):
        mask = get_mask_from_lengths(lens // self.n_group_size)
        mask = mask[:, None].float()
        loss_dict = {}
        if "z" in model_output:
            n_elements = lens.sum() // self.n_group_size
            n_dims = model_output["z"].size(1)

            loss, loss_prior = compute_flow_loss(
                model_output["z"],
                model_output["log_det_W_list"],
                model_output["log_s_list"],
                n_elements,
                n_dims,
                mask,
                self.sigma,
            )
            loss_dict = {
                "loss_{}".format(self.name): (loss, self.loss_weight),
                "loss_prior_{}".format(self.name): (loss_prior, 0.0),
            }
        elif "x_hat" in model_output:
            loss_dict = compute_regression_loss(
                model_output["x_hat"], model_output["x"], mask, self.name
            )
            for k, v in loss_dict.items():
                loss_dict[k] = (v, self.loss_weight)

        if len(loss_dict) == 0:
            raise Exception("loss not supported")

        return loss_dict


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0), value=self.blank_logprob
        )
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : query_lens[bid], :, : key_lens[bid] + 1
            ]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            cost_total += ctc_cost
        cost = cost_total / attn_logprob.shape[0]
        return cost


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        return -log_sum / hard_attention.sum()


class RADTTSLoss(torch.nn.Module):
    def __init__(
        self,
        sigma=1.0,
        n_group_size=1,
        dur_model_config=None,
        f0_model_config=None,
        energy_model_config=None,
        vpred_model_config=None,
        loss_weights=None,
    ):
        super(RADTTSLoss, self).__init__()
        self.sigma = sigma
        self.n_group_size = n_group_size
        self.loss_weights = loss_weights
        self.attn_ctc_loss = AttentionCTCLoss(
            blank_logprob=loss_weights.get("blank_logprob", -1)
        )
        self.loss_fns = {}
        if dur_model_config is not None:
            self.loss_fns["duration_model_outputs"] = AttributePredictionLoss(
                "duration", dur_model_config, loss_weights["dur_loss_weight"]
            )

        if f0_model_config is not None:
            self.loss_fns["f0_model_outputs"] = AttributePredictionLoss(
                "f0", f0_model_config, loss_weights["f0_loss_weight"], sigma=1.0
            )

        if energy_model_config is not None:
            self.loss_fns["energy_model_outputs"] = AttributePredictionLoss(
                "energy", energy_model_config, loss_weights["energy_loss_weight"]
            )

        if vpred_model_config is not None:
            self.loss_fns["vpred_model_outputs"] = AttributePredictionLoss(
                "vpred", vpred_model_config, loss_weights["vpred_loss_weight"]
            )

    def forward(self, model_output, in_lens, out_lens):
        loss_dict = {}
        if len(model_output["z_mel"]):
            n_elements = out_lens.sum() // self.n_group_size
            mask = get_mask_from_lengths(out_lens // self.n_group_size)
            mask = mask[:, None].float()
            n_dims = model_output["z_mel"].size(1)
            loss_mel, loss_prior_mel = compute_flow_loss(
                model_output["z_mel"],
                model_output["log_det_W_list"],
                model_output["log_s_list"],
                n_elements,
                n_dims,
                mask,
                self.sigma,
            )
            loss_dict["loss_mel"] = (loss_mel, 1.0)  # loss, weight
            loss_dict["loss_prior_mel"] = (loss_prior_mel, 0.0)

        ctc_cost = self.attn_ctc_loss(model_output["attn_logprob"], in_lens, out_lens)
        loss_dict["loss_ctc"] = (ctc_cost, self.loss_weights["ctc_loss_weight"])

        for k in model_output:
            if k in self.loss_fns:
                if model_output[k] is not None and len(model_output[k]) > 0:
                    t_lens = in_lens if "dur" in k else out_lens
                    mout = model_output[k]
                    for loss_name, v in self.loss_fns[k](mout, t_lens).items():
                        loss_dict[loss_name] = v

        return loss_dict
