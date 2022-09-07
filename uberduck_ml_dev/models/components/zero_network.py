from torch import nn
import torch

# NOTE (Sam): this is a hack that enables torchscipt compilation with n_speakers = 1 and has_speaker_embedding=True
class ZeroNetwork(nn.Module):
    def forward(self, x):

        return torch.zeros_like(x)
