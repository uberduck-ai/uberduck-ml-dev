import torch
from torch import nn

from .data.batch import Batch

# NOTE (Sam): This should get its own file, and loss should get its own class.
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
