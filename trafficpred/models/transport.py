"""Distribution heads that move mass instead of rescoring cells.

TransportHead turns per-cell logits into a local transport plan: every cell sends its share of
traffic to the cells within `radius` of it (a softmax over those offsets, so each cell's mass is
conserved exactly; offsets outside the grid are masked). The next distribution is the last one
pushed through the plan, and the plans are chained over the output steps, so step t can reach
t * radius cells away:

    P_1 = move_1(P_last),  P_t = move_t(P_{t-1})

With periodic inputs (--periods day week) every target step also has "the same time one period
back" for each period. Each of those is moved by its own plan, and the candidates are mixed per
cell with learned weights, then renormalised:

    P_t = normalise( sum_s w_{t,s} * move_{t,s}(source_s) ),  sources = P_{t-1}, P_{t-day}, P_{t-week}

At initialisation every plan keeps ~94% of a cell's mass in place and the mixture is uniform.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STAY_LOGIT = 6.0   # initial logit of "stay put" (others 0): ~94% stays with radius 2
FLOOR = 1e-7       # keeps every cell's probability above 0 (cells nothing can reach)


class TransportHead(nn.Module):
    """Maps features (B, F, H, W) and the input distributions to the output distributions."""

    def __init__(self, T_out, C, radius, n_periodic=0):
        super().__init__()
        self.T_out, self.C, self.r = T_out, C, radius
        self.K = (2 * radius + 1) ** 2
        self.S = 1 + n_periodic                    # sources: the previous step + one per period
        self.mix = self.S > 1
        self.channels = T_out * C * self.S * (self.K + (1 if self.mix else 0))

    def init_head(self, weight, bias):
        """Zero weights; bias favouring "stay" for every plan (the bias layout matches split())."""
        nn.init.zeros_(weight)
        with torch.no_grad():
            b = torch.zeros(self.T_out, self.C, self.S, self.K + (1 if self.mix else 0))
            b[..., self.K // 2] = STAY_LOGIT
            bias.copy_(b.reshape(-1).repeat(bias.numel() // b.numel()))

    def plans(self, out, H, W):
        """(B, channels, H, W) -> move weights (B, T, C, S, K, H*W), mixture logits (B, T, C, S, H*W)."""
        B = out.shape[0]
        out = out.reshape(B, self.T_out, self.C, self.S, -1, H * W)
        move, mix = out[:, :, :, :, :self.K], out[:, :, :, :, self.K:]
        valid = F.unfold(torch.ones(1, 1, H, W, device=out.device, dtype=out.dtype),
                         2 * self.r + 1, padding=self.r) > 0          # (1, K, H*W): offset in the grid
        move = move.masked_fill(~valid[:, None, None, None], float("-inf")).softmax(dim=4)
        return move, (mix[:, :, :, :, 0] if self.mix else None)

    def move(self, p, w, H, W):
        """Push p (B, C, H*W) through plan w (B, C, K, H*W): cell l sends p[l] * w[k, l] to l + offset k."""
        B, C = p.shape[:2]
        sent = (p[:, :, None] * w).reshape(B, C * self.K, H * W)
        out = F.fold(sent.reshape(B * C, self.K, H * W), (H, W), 2 * self.r + 1, padding=self.r)
        return out.reshape(B, C, H * W)

    def forward(self, out, p_in, H, W):
        """out: head output (B, channels, H, W); p_in: input distributions (B, T_in, C, H, W) with the
        periodic frames first (period j, step t at index j * T_out + t) and the most recent frame last."""
        B = out.shape[0]
        move, mix = self.plans(out, H, W)
        p_flat = p_in.reshape(B, p_in.shape[1], self.C, H * W)
        prev, steps = p_flat[:, -1], []
        for t in range(self.T_out):
            cands = [self.move(prev, move[:, t, :, 0], H, W)]
            for j in range(1, self.S):
                cands.append(self.move(p_flat[:, (j - 1) * self.T_out + t], move[:, t, :, j], H, W))
            if self.mix:
                weights = mix[:, t].softmax(dim=2)                     # (B, C, S, H*W), per cell
                p = (weights * torch.stack(cands, dim=2)).sum(dim=2)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                p = cands[0]
            p = (1 - FLOOR * H * W) * p + FLOOR
            steps.append(p)
            prev = p
        return torch.stack(steps, dim=1).reshape(B, self.T_out, self.C, H, W)
