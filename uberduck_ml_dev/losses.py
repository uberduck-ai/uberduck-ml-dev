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
