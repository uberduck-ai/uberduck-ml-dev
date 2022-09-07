from torch import nn
from torch.nn import functional as F
from ..common import LinearNorm


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.dropout_rate = 0.5

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout_rate, training=True)
        return x
