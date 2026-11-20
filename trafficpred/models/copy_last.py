import torch.nn as nn


class CopyLast(nn.Module):
    def __init__(self, t_out, channels=None):
        """channels: how many of the input channels are flow (the rest, e.g. --od maps, aren't copied)."""
        super().__init__()
        self.t_out, self.channels = t_out, channels

    def forward(self, x):
        x = x[:, -1:, :self.channels] if self.channels else x[:, -1:]
        return x.expand(-1, self.t_out, *x.shape[2:])
