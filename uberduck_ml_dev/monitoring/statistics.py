__all__ = ["get_alignment_metrics"]

import torch
from ..utils.utils import get_mask_from_lengths


def get_alignment_metrics(
    alignments, average_across_batch=True, input_lengths=None, output_lengths=None
):
    alignments = alignments.transpose(1, 2)  # [B, dec, enc] -> [B, enc, dec]
    if input_lengths == None:
        input_lengths = torch.ones(alignments.size(0), device=alignments.device) * (
            alignments.shape[1] - 1
        )  # [B] # 147
    if output_lengths == None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device) * (
            alignments.shape[2] - 1
        )  # [B] # 767

    batch_size = alignments.size(0)
    optimums = torch.sqrt(
        input_lengths.double().pow(2) + output_lengths.double().pow(2)
    ).view(batch_size)

    # [B, enc, dec] -> [B, dec], [B, dec]
    values, cur_idxs = torch.max(alignments, 1)

    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:, 0][:, None], cur_idxs[:, :-1]), dim=1)
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5)  # [B, dec]
    dist.masked_fill_(
        ~get_mask_from_lengths(output_lengths, max_len=dist.size(1)), 0.0
    )  # set dist of padded to zero
    dist = dist.sum(dim=(1))  # get total dist for each B
    diagonalness = (dist + 1.4142135) / optimums  # dist / optimal dist

    maxes = alignments.max(axis=1)[0].mean(axis=1)
    if average_across_batch:
        diagonalness = diagonalness.mean()
        maxes = maxes.mean()

    output = {}
    output["diagonalness"] = diagonalness
    output["max"] = maxes

    return output
