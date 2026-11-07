import torch.nn as nn


class CopyLast(nn.Module):
    def __init__(self, t_out):
        super().__init__()
        self.t_out = t_out

    def forward(self, x):
        return x[:, -1:].expand(-1, self.t_out, *x.shape[2:])
