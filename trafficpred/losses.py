import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.sinkhorn import sinkhorn_divergence, split_flow


def masked_l1_loss(pred, y):
    mask = (y != 0).float()
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)


class PointwiseLoss(nn.Module):
    needs_parts = False
    FUNCTIONS = {"mse": F.mse_loss, "mae": F.l1_loss, "masked_mae": masked_l1_loss}

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.fn = self.FUNCTIONS[kind]

    def forward(self, pred, y):
        loss = self.fn(pred.float(), y)
        return loss, {self.kind: loss.detach()}


class DistVolLoss(nn.Module):
    needs_parts = True

    def __init__(self, l1=1.0, l2=10.0, l3=1.0, eps=0.01, iters=30):
        super().__init__()
        self.l1, self.l2, self.l3, self.eps, self.iters = l1, l2, l3, eps, iters

    def forward(self, parts, y):
        x_hat, p_hat, n_hat = (t.float() for t in parts)
        H, W = y.shape[-2:]
        p, n = split_flow(y.clamp_min(0))
        reconstruction = (x_hat - y).abs().mean()
        transport = torch.zeros((), device=y.device)
        if self.l2 > 0:
            transport = sinkhorn_divergence(p_hat.reshape(-1, H, W), p.reshape(-1, H, W),
                                            eps=self.eps, iters=self.iters).mean()
        volume = (n_hat - n).abs().mean() / (H * W)
        total = self.l1 * reconstruction + self.l2 * transport + self.l3 * volume
        return total, {"l1": reconstruction.detach(), "sinkhorn": transport.detach(),
                       "volume": volume.detach()}


LOSSES = ("mse", "mae", "masked_mae", "sinkhorn")


def make_loss(name, l1=1.0, l2=10.0, l3=1.0, eps=0.01, sinkhorn_iters=30):
    if name == "sinkhorn":
        return DistVolLoss(l1, l2, l3, eps, sinkhorn_iters)
    return PointwiseLoss(name)
