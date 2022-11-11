__all__ = ['fix_len_compatibility_text_edit', 'EdiTTS']


import random
import math

from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
import monotonic_align

from ..vendor.tfcompat.hparam import HParams
from ..text.symbols import SYMBOL_SETS
from ..text.util import text_to_sequence, text_to_sequence_for_editts
from ..utils.utils import intersperse, intersperse_emphases
from .gradtts import (
    GradTTS,
    sequence_mask,
    generate_path,
    fix_len_compatibility,
    get_noise,
    DEFAULTS,
)




def fix_len_compatibility_text_edit(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length -= 1


class EdiTTS(torch.nn.Module):
    def __init__(self, gradtts_model):
        super(EdiTTS, self).__init__()
        self.gradtts_model = gradtts_model

    def infer_edit_content(
        self,
        text1,
        text2,
        n_timesteps=10,
        symbol_set="gradtts",
        temperature=1.0,
        stoc=False,
        length_scale=1.0,
        soften_mask=True,
        n_soften_text=9,
        n_soften=16,
        amax=0.9,
        amin=0.1,
        intersperse_token=148,
    ):
        """
        EdiTTS
        Edit speech/audio via content substitution.
        This function will substitute the desired portion of text2 into the specified location of text1.

        Arguments:
        text1 (str): text to substitute content in to. e.g. "This is a | blue | pencil"
        text2 (str): text to substitute audio from. e.g. "This is a | red | pen."
        n_timesteps (int): number of steps to use for reverse diffusion in decoder.
        symbol_set (str): symbol set key to lookup the symbol set
        intersperse_token (int): value used for interspersing

        Output:
        y_dec1: Mel spectrogram of text1
        y_dec2: Mel spectrogram of text2
        y_dec_edit: Mel spectrogram of source of text2 substituted in to text1 via EdiTTS
        y_dec_cat: Mel spectrogram of source of text2 substituted in to text1 via mel concatenation

        Usage:
        y_dec1, y_dec2, y_dec_edit, y_dec_cat = model.infer_edit_content("This is a | blue | pencil.",
                                                                    "This is a | red | pen.",
                                                                    n_timesteps=10,
                                                                    symbol_set="gradtts")
        y_dec1: "this is a blue pencil"
        y_dec2: "this is a red pen"
        y_dec_edit: "this is a red pencil" (EdiTTS)
        y_dec_cat: "this is a red pencil" (Mel concatenation)

        """
        sequence1, emphases1 = text_to_sequence_for_editts(
            text1, cleaner_names=["english_cleaners"], symbol_set=symbol_set
        )
        sequence2, emphases2 = text_to_sequence_for_editts(
            text2, cleaner_names=["english_cleaners"], symbol_set=symbol_set
        )
        x1 = torch.LongTensor(intersperse(sequence1, intersperse_token)).cuda()[None]
        x2 = torch.LongTensor(intersperse(sequence2, intersperse_token)).cuda()[None]
        emphases1 = intersperse_emphases(emphases1)
        emphases2 = intersperse_emphases(emphases2)
        x_lengths1 = torch.LongTensor([x1.shape[-1]]).cuda()
        x_lengths2 = torch.LongTensor([x2.shape[-1]]).cuda()

        y_dec1, y_dec2, y_dec_edit, y_dec_cat = self.edit_content(
            x1,
            x2,
            x_lengths1,
            x_lengths2,
            emphases1,
            emphases2,
            n_timesteps,
            temperature=temperature,
            stoc=stoc,
            length_scale=length_scale,
            soften_mask=soften_mask,
            n_soften_text=n_soften_text,
            n_soften=n_soften,
            amax=amax,
            amin=amin,
        )
        return y_dec1, y_dec2, y_dec_edit, y_dec_cat

    def infer_edit_content_with_source(
        self,
        mel,
        sub_time_start,
        sub_time_stop,
        text,
        n_timesteps=50,
        desired_time=None,
        symbol_set="gradtts",
        temperature=1.0,
        stoc=False,
        length_scale=1.0,
        soften_mask=True,
        n_soften_text=9,
        n_soften=16,
        amax=0.9,
        amin=0.1,
        intersperse_token=148,
    ):
        """
        EdiTTS
        Edit speech/audio via content substitution.
        This function will substitute the specified time region of mel with the portion of text surrounded by |.

        Arguments:
        mel (torch.Tensor): text to generate and use for editing content . e.g. "This is a | blue | pencil"
        sub_time_start (float): Starting time for substitution in mel (seconds)
        sub_time_stop (float): Ending time for substitution in mel (seconds)
        n_timesteps (int): number of steps to use for reverse diffusion in decoder
        symbol_set (str): symbol set key to lookup the symbol set
        desired_time (float): Length of time for audio to be substituted into mel (seconds)
        intersperse_token (int): value used for interspersing

        Output:
        y_dec1: Mel spectrogram of text1
        y_dec2: Mel spectrogram of text2
        y_dec_edit: Mel spectrogram of source of text2 substituted in to text1 via EdiTTS
        y_dec_cat: Mel spectrogram of source of text2 substituted in to text1 via mel concatenation

        """
        sequence, emphases = text_to_sequence_for_editts(
            text, cleaner_names=["english_cleaners"], symbol_set=symbol_set
        )
        x = torch.LongTensor(intersperse(sequence, intersperse_token)).cuda()[None]
        emphases = intersperse_emphases(emphases)
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

        y_dec1, y_dec2, y_dec_edit, y_dec_cat = self.edit_content(
            x2=x,
            x2_lengths=x_lengths,
            emphases2=emphases,
            n_timesteps=n_timesteps,
            mel1=mel,
            i1=sub_time_start,
            j1=sub_time_stop,
            desired_time=desired_time,
            temperature=temperature,
            stoc=stoc,
            length_scale=length_scale,
            soften_mask=soften_mask,
            n_soften_text=n_soften_text,
            n_soften=n_soften,
            amax=amax,
            amin=amin,
        )
        return y_dec1, y_dec2, y_dec_edit, y_dec_cat

    @torch.no_grad()
    def edit_content(
        self,
        x1=None,
        x2=None,
        x1_lengths=None,
        x2_lengths=None,
        emphases1=None,
        emphases2=None,
        n_timesteps=50,
        mel1=None,
        i1=None,
        j1=None,
        desired_time=None,
        temperature=1.0,
        stoc=False,
        length_scale=1.0,
        soften_mask=True,
        n_soften_text=9,
        n_soften=16,
        amax=0.9,
        amin=0.1,
    ):
        def _process_input(x, x_lengths):
            x, x_lengths = self.gradtts_model.relocate_input([x, x_lengths])

            # encoded_text, durations, text_mask
            mu_x, logw, x_mask = self.gradtts_model.encoder(x, x_lengths)
            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w) * length_scale

            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = int(y_lengths.max())
            y_max_length_ = fix_len_compatibility(y_max_length)

            y_mask = (
                sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
            )
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)  # [1, n_mels, T]
            return mu_y, attn, y_mask, y_max_length, y_lengths

        def _process_input_time_constraint(x, x_lengths, emphases, desired_time):
            x, x_lengths = self.gradtts_model.relocate_input([x, x_lengths])

            # encoded_text, durations, text_mask
            mu_x, logw, x_mask = self.gradtts_model.encoder(x, x_lengths)
            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w)

            # Add time constraint
            w_slice = w_ceil.squeeze()[emphases[0][0] : emphases[0][1]]
            time_scale = (
                (desired_time * self.gradtts_model.sampling_rate)
                / self.gradtts_model.hop_length
            ) / torch.sum(w_slice)
            w_ceil = w_ceil * time_scale

            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = int(y_lengths.max())
            y_max_length_ = fix_len_compatibility(y_max_length)

            y_mask = (
                sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
            )
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)  # [1, n_mels, T]
            return mu_y, attn, y_mask, y_max_length, y_lengths

        def _soften_juntions(
            y_edit, y1, y2, y_edit_lengths, y1_lengths, y2_lengths, i1, j1, i2, j2
        ):
            for n in range(1, n_soften_text + 1):
                alpha = (amax - amin) * (n_soften_text - n) / (n_soften_text - 1) + amin
                if i1 - n >= 0 and i2 - n >= 0:
                    y_edit[:, :, i1 - n] = (1 - alpha) * y1[:, :, i1 - n] + alpha * y2[
                        :, :, i2 - n
                    ]
                if (
                    i1 + (j2 - i2) + n < y_edit_lengths
                    and j1 + (n - 1) < y1_lengths
                    and j2 + (n - 1) < y2_lengths
                ):
                    y_edit[:, :, i1 + (j2 - i2) + (n - 1)] = (1 - alpha) * y1[
                        :, :, j1 + (n - 1)
                    ] + alpha * y2[:, :, j2 + (n - 1)]
            return y_edit

        if mel1 is not None:
            assert len(x2) == 1 and x1 is None
            assert emphases2 is not None
            assert len(emphases2) == 1
        else:
            assert len(x1) == 1 and len(x2) == 1
            assert emphases1 is not None and emphases2 is not None
            assert len(emphases1) == 1 and len(emphases2) == 1

        if mel1 is not None:
            mu_y1 = mel1
            y1_max_length = mel1.shape[-1]
            y1_lengths = torch.LongTensor([mel1.shape[-1]]).cuda()
        else:
            mu_y1, attn1, y1_mask, y1_max_length, y1_lengths = _process_input(
                x1, x1_lengths
            )  # mu_y1: [1, n_mels, T]

        if desired_time:
            (
                mu_y2,
                attn2,
                y2_mask,
                y2_max_length,
                y2_lengths,
            ) = _process_input_time_constraint(
                x2,
                x2_lengths,
                emphases2,
                desired_time,
            )  # mu_y2: [1, n_mels, T]
        else:
            mu_y2, attn2, y2_mask, y2_max_length, y2_lengths = _process_input(
                x2, x2_lengths
            )  # mu_y2: [1, n_mels, T]

        if not i1 and not j1:
            attn1 = attn1.squeeze()  # [N, T]
            i1 = (
                attn1[: emphases1[0][0]].sum().long().item()
                if emphases1[0][0] > 0
                else 0
            )
            j1 = attn1[: emphases1[0][1]].sum().long().item()

        attn2 = attn2.squeeze()  # [N, T]
        i2 = attn2[: emphases2[0][0]].sum().long().item() if emphases2[0][0] > 0 else 0
        j2 = attn2[: emphases2[0][1]].sum().long().item()

        # Step 1. Direct concatenation
        mu_y1_a, mu_y1_c = mu_y1[:, :, :i1], mu_y1[:, :, j1:y1_lengths]
        mu_y2_b = mu_y2[:, :, i2:j2]
        mu_y_edit = torch.cat((mu_y1_a, mu_y2_b, mu_y1_c), dim=2)
        y_edit_lengths = int(mu_y_edit.shape[2])

        # Step 2. Soften junctions
        mu_y_edit = _soften_juntions(
            mu_y_edit,
            mu_y1,
            mu_y2,
            y_edit_lengths,
            y1_lengths,
            y2_lengths,
            i1,
            j1,
            i2,
            j2,
        )

        y_edit_length_ = fix_len_compatibility_text_edit(y_edit_lengths)
        y_edit_lengths_tensor = torch.tensor([y_edit_lengths]).long().to(x2.device)
        y_edit_mask_for_scorenet = (
            sequence_mask(y_edit_lengths_tensor, y_edit_length_)
            .unsqueeze(1)
            .to(mu_y1.dtype)
        )

        eps1 = torch.randn_like(mu_y1, device=mu_y1.device) / temperature
        eps2 = torch.randn_like(mu_y2, device=mu_y1.device) / temperature
        eps_edit = torch.cat(
            (eps1[:, :, :i1], eps2[:, :, i2:j2], eps1[:, :, j1:y1_lengths]), dim=2
        )
        z1 = mu_y1 + eps1
        z2 = mu_y2 + eps2
        z_edit = mu_y_edit + eps_edit

        if z_edit.shape[2] < y_edit_length_:
            pad = y_edit_length_ - z_edit.shape[2]
            zeros = torch.zeros_like(z_edit[:, :, :pad])
            z_edit = torch.cat((z_edit, zeros), dim=2)
            mu_y_edit = torch.cat((mu_y_edit, zeros), dim=2)
        elif z_edit.shape[2] > y_edit_length_:
            res = z_edit.shape[2] - y_edit_length_
            z_edit = z_edit[:, :, :-res]
            mu_y_edit = mu_y_edit[:, :, :-res]

        y_edit_mask_for_gradient = torch.zeros_like(mu_y_edit[:, :1, :])
        y_edit_mask_for_gradient[:, :, i1 : i1 + (j2 - i2)] = 1

        if mel1 is not None:
            dec1 = mu_y1
        else:
            dec1 = self.gradtts_model.decoder(z1, y1_mask, mu_y1, n_timesteps, stoc)

        dec2, dec_edit = self.double_forward_text(
            z2,
            z_edit,
            mu_y2,
            mu_y_edit,
            y2_mask,
            y_edit_mask_for_scorenet,
            y_edit_mask_for_gradient,
            i1,
            j1,
            i2,
            j2,
            n_timesteps,
            stoc,
            soften_mask,
            n_soften,
        )

        dec1 = dec1[:, :, :y1_max_length]
        dec2 = dec2[:, :, :y2_max_length]
        dec_edit = dec_edit[:, :, :y_edit_lengths]
        dec_cat = torch.cat(
            (dec1[:, :, :i1], dec2[:, :, i2:j2], dec1[:, :, j1:y1_lengths]), dim=2
        )

        return dec1, dec2, dec_edit, dec_cat

    @torch.no_grad()
    def double_forward_text(
        self,
        z,
        z_edit,
        mu,
        mu_edit,
        mask,
        mask_edit_net,
        mask_edit_grad,
        i1,
        j1,
        i2,
        j2,
        n_timesteps,
        stoc=False,
        soften_mask=True,
        n_soften=20,
    ):
        if soften_mask:
            kernel = [
                2 ** ((n_soften - 1) - abs(n_soften - 1 - i))
                for i in range(2 * n_soften - 1)
            ]  # [1, 2, 4, ..., 2^n_soften , 2^(n_soften-1), ..., 2, 1]
            kernel = [i / sum(kernel[: len(kernel) // 2 + 1]) for i in kernel]
            w = (
                torch.tensor(kernel)
                .view(1, 1, 1, len(kernel))
                .to(mask_edit_grad.device)
                .float()
            )
            mask_edit_soft = mask_edit_grad.unsqueeze(1).contiguous()
            mask_edit_soft = F.pad(
                mask_edit_soft,
                (len(kernel) // 2, len(kernel) // 2, 0, 0),
                mode="replicate",
            )
            mask_edit_soft = F.conv2d(
                mask_edit_soft,
                w,
                bias=None,
                stride=1,
            )
            mask_edit_soft = mask_edit_soft.squeeze(1)
            mask_edit_grad = mask_edit_grad + (1 - mask_edit_grad) * mask_edit_soft

        h = 1.0 / n_timesteps
        xt = z * mask
        xt_edit = z_edit * mask_edit_net

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(
                time,
                self.gradtts_model.decoder.beta_min,
                self.gradtts_model.decoder.beta_max,
                cumulative=False,
            )
            if stoc:  # adds stochastic term
                # NOTE: should not come here
                assert False
                dxt_det = 0.5 * (mu - xt) - self.gradtts_model.decoder.estimator(
                    xt, mask, mu, t
                )
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(
                    z.shape, dtype=z.dtype, device=z.device, requires_grad=False
                )
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (
                    mu - xt - self.gradtts_model.decoder.estimator(xt, mask, mu, t)
                )
                dxt = dxt * noise_t * h
                dxt_edit = 0.5 * (
                    mu_edit
                    - xt_edit
                    - self.gradtts_model.decoder.estimator(
                        xt_edit, mask_edit_net, mu_edit, t
                    )
                )
                dxt_edit = dxt_edit * noise_t * h

            xt = (xt - dxt) * mask

            dxt_trg = torch.zeros_like(dxt_edit)
            dxt_trg[:, :, i1 : i1 + (j2 - i2)] = dxt[:, :, i2:j2]

            xt_edit = (
                xt_edit - (mask_edit_grad * dxt_trg + (1 - mask_edit_grad) * dxt_edit)
            ) * mask_edit_net

        return xt, xt_edit
