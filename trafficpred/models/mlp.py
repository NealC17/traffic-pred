import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, t_in=4, t_out=4, channels=2, height=32, width=32, hidden=512, depth=3,
                 dropout=0.1):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        layers, d = [], t_in * channels * height * width
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, t_out * channels * height * width))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.net(x.reshape(B, -1)).view(B, self.t_out, C, H, W)
