"""
STDCformer: A Transformer-Based Model with a Spatial-Temporal Causal
De-Confounding Strategy for Crowd Flow Prediction
(He, Shen, Xu, Luo, Li — arXiv:2412.02942 / Information Fusion 2026).

Unofficial PyTorch implementation of the E-M-D skeleton:

    X --E--> H_past --M--> H_future --D--> Y_hat

  E  (Encoder)  : data embedding + Spatial-Temporal Embedding (STE) +
                  stack of Spatial-Temporal De-Confounded (STDC) attention blocks
  M  (Mapping)  : Cross-Time Attention (CTA) — Q = STE_future, K = STE_past,
                  V = H_past (Eq. 7)
  D  (Decoder)  : stack of STDC attention blocks on the future representation
                  + Conv2d prediction head (Sec. 3.5.2)

Each STDC block runs spatial attention (across regions, per time step, Eq. 4)
and temporal attention (across time steps, per region, Eq. 5) in parallel and
fuses them with a learned back-door-adjustment gate (Eq. 6):

    P(C_S) = sigmoid(C_S + C_T),  P(C_T) = 1 - P(C_S)
    H      = P(C_S) * STR'_S + P(C_T) * STR'_T

Choices for things the paper leaves unspecified are marked "[impl choice]".
Tensor layout everywhere: (batch B, time T, regions N, channels).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class STDCConfig:
    num_nodes: int                     # n regions
    in_dim: int = 2                    # f: inflow, outflow
    out_dim: int = 2
    input_len: int = 6                 # T_p
    output_len: int = 6                # T_f
    d_model: int = 64                  # d
    num_heads: int = 8
    enc_layers: int = 5                # 5 for MHT, 6 for BKL in the paper
    dec_layers: int = 5                # [impl choice] paper uses "same structure"
    cta_layers: int = 1                # [impl choice] not stated in paper
    ffn_mult: int = 4                  # [impl choice]
    dropout: float = 0.1               # [impl choice]
    # --- spatial confounders S_i ---
    lap_dim: int = 8                   # d_lap (paper: 8, searched {6,8,10,12})
    spatial_feat_dim: int = 0          # s: static region features (POI, housing, crime...)
    use_node_id: bool = True           # taxi-zone ID embedding
    # --- temporal confounders T_j ---
    steps_per_day: int = 24            # 1-hour intervals
    use_time_of_day: bool = True       # False when the data has no timestamps (falls back to
                                       # learned embeddings of the step's position in the window)
    use_day_of_week: bool = True
    use_holiday: bool = True
    temporal_feat_dim: int = 0         # t: continuous exogenous (temperature, weather...)


# ---------------------------------------------------------------------------
# Spatial-Temporal Embedding (Sec. 3.2, Eq. 3)
# ---------------------------------------------------------------------------
class SpatialConfounderEncoder(nn.Module):
    """C_S = Conv(S ⊕ Laplacian) -> (N, d). 1x1 conv == per-token linear."""

    def __init__(self, cfg: STDCConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        in_dim = cfg.lap_dim + cfg.spatial_feat_dim
        self.node_emb = nn.Embedding(cfg.num_nodes, d) if cfg.use_node_id else None
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d)) if in_dim > 0 else None
        assert self.node_emb is not None or self.proj is not None, "need some spatial confounder"

    def forward(self, lap_pe: torch.Tensor | None, spatial_feat: torch.Tensor | None) -> torch.Tensor:
        out = 0
        if self.proj is not None:
            parts = [p for p in (lap_pe, spatial_feat) if p is not None and p.shape[-1] > 0]
            out = out + self.proj(torch.cat(parts, dim=-1))
        if self.node_emb is not None:
            out = out + self.node_emb.weight
        return out  # (N, d)


class TemporalConfounderEncoder(nn.Module):
    """C_T = Conv(T) -> (B, T, d). Categorical attributes use embeddings."""

    def __init__(self, cfg: STDCConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.tod = nn.Embedding(cfg.steps_per_day, d) if cfg.use_time_of_day else None
        # Fallback without timestamps: one embedding per window position
        # (0..T_p-1 = past steps, T_p..T_p+T_f-1 = future steps).
        self.pos = None if cfg.use_time_of_day else nn.Embedding(cfg.input_len + cfg.output_len, d)
        self.dow = nn.Embedding(7, d) if cfg.use_day_of_week else None
        self.hol = nn.Embedding(2, d) if cfg.use_holiday else None
        self.exo = (nn.Sequential(nn.Linear(cfg.temporal_feat_dim, d), nn.ReLU(), nn.Linear(d, d))
                    if cfg.temporal_feat_dim > 0 else None)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(d, d))

    def forward(self, tod, dow=None, hol=None, exo=None) -> torch.Tensor:
        # `tod` holds time-of-day indices, or window positions when use_time_of_day=False
        h = self.tod(tod) if self.tod is not None else self.pos(tod)
        if self.dow is not None and dow is not None:
            h = h + self.dow(dow)
        if self.hol is not None and hol is not None:
            h = h + self.hol(hol)
        if self.exo is not None and exo is not None:
            h = h + self.exo(exo)
        return self.out(h)  # (B, T, d)


class STEmbedding(nn.Module):
    """STE = Conv(C_S) + Conv(C_T), broadcast to (B, T, N, d)."""

    def __init__(self, cfg: STDCConfig):
        super().__init__()
        d = cfg.d_model
        self.spatial = SpatialConfounderEncoder(cfg)
        self.temporal = TemporalConfounderEncoder(cfg)
        self.fs = nn.Linear(d, d)
        self.ft = nn.Linear(d, d)

    def confounders(self, lap_pe, spatial_feat, tod, dow, hol, exo):
        c_s = self.spatial(lap_pe, spatial_feat)          # (N, d)
        c_t = self.temporal(tod, dow, hol, exo)            # (B, T, d)
        return c_s, c_t

    def forward(self, c_s, c_t):
        # (B, T, 1, d) + (1, 1, N, d)
        return self.ft(c_t).unsqueeze(2) + self.fs(c_s)[None, None]


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Scaled dot-product MHA. Q/K/V project from separate inputs so it can be
    reused for self-attention (STDC block) and cross-time attention (CTA)."""

    def __init__(self, q_dim, k_dim, v_dim, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dh = num_heads, d_model // num_heads
        self.wq = nn.Linear(q_dim, d_model)
        self.wk = nn.Linear(k_dim, d_model)
        self.wv = nn.Linear(v_dim, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = dropout

    def forward(self, q, k, v):
        # q: (..., Lq, q_dim)   k, v: (..., Lk, *)
        *lead, Lq, _ = q.shape
        Lk = k.shape[-2]
        Q = self.wq(q).view(*lead, Lq, self.h, self.dh).transpose(-2, -3)
        K = self.wk(k).view(*lead, Lk, self.h, self.dh).transpose(-2, -3)
        V = self.wv(v).view(*lead, Lk, self.h, self.dh).transpose(-2, -3)
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(-2, -3).reshape(*lead, Lq, self.h * self.dh)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d, mult, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Spatial-Temporal De-Confounded attention block (Sec. 3.3.3, Eqs. 4–6)
# ---------------------------------------------------------------------------
class STDCBlock(nn.Module):
    """
    Input  STR = [H ⊕ STE] in R^{2d}  (W^{Q,K,V} in R^{2d x d}, per the paper)
    Output H'  in R^{d}

    [impl choice] The paper's Eq. 3 adds V' + STE but the text and the
    W in R^{2d x d} shapes imply concatenation. We concatenate, and re-inject
    STE at every block so every layer sees the confounders.
    [impl choice] Pre-norm residual + FFN (paper does not describe them).
    """

    def __init__(self, cfg: STDCConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_h = nn.LayerNorm(d)
        self.norm_ste = nn.LayerNorm(d)
        self.spatial_attn = MultiHeadAttention(2 * d, 2 * d, 2 * d, d, cfg.num_heads, cfg.dropout)
        self.temporal_attn = MultiHeadAttention(2 * d, 2 * d, 2 * d, d, cfg.num_heads, cfg.dropout)
        # Back-door gate: P(C_S) = sigmoid(C_S + C_T), learned per block, per channel.
        self.gate_s = nn.Linear(d, d)
        self.gate_t = nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = FeedForward(d, cfg.ffn_mult, cfg.dropout)

    def forward(self, h, ste, c_s, c_t, return_gate=False):
        # h, ste: (B, T, N, d); c_s: (N, d); c_t: (B, T, d)
        x = torch.cat([self.norm_h(h), self.norm_ste(ste)], dim=-1)          # (B,T,N,2d)

        # Spatial attention: for each time step T_j, attend across N regions.
        str_s = self.spatial_attn(x, x, x)                                   # (B,T,N,d)
        # Temporal attention: for each region S_i, attend across T steps.
        xt = x.transpose(1, 2)                                               # (B,N,T,2d)
        str_t = self.temporal_attn(xt, xt, xt).transpose(1, 2)               # (B,T,N,d)

        # De-confounded fusion (Eq. 6) — approximates the back-door adjustment
        # P(Y|do(X)) = Σ_c P(Y|X,c) P(c) over c ∈ {C_S, C_T}.
        p_cs = torch.sigmoid(self.gate_s(c_s)[None, None] + self.gate_t(c_t).unsqueeze(2))
        fused = p_cs * str_s + (1.0 - p_cs) * str_t

        h = h + self.drop(fused)
        h = h + self.ffn(self.norm_ffn(h))
        return (h, p_cs) if return_gate else h


# ---------------------------------------------------------------------------
# Cross-Time Attention mapping M (Sec. 3.4, Eq. 7)
# ---------------------------------------------------------------------------
class CrossTimeAttention(nn.Module):
    """Per region: Q = STE_future W_Q, K = STE_past W_K, V = H_past W_V.
    Attention map per region has shape (T_f, T_p)."""

    def __init__(self, cfg: STDCConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_q = nn.LayerNorm(d)
        self.norm_k = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, d, d, d, cfg.num_heads, cfg.dropout)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = FeedForward(d, cfg.ffn_mult, cfg.dropout)

    def forward(self, h_past, ste_past, ste_future, h_future=None):
        # all inputs (B, T*, N, d) -> per-region sequences (B, N, T*, d)
        q = self.norm_q(ste_future).transpose(1, 2)
        k = self.norm_k(ste_past).transpose(1, 2)
        v = self.norm_v(h_past).transpose(1, 2)
        out = self.attn(q, k, v).transpose(1, 2)                             # (B,T_f,N,d)
        # [impl choice] residual onto the previous future estimate for stacked CTA layers
        h = out if h_future is None else h_future + out
        return h + self.ffn(self.norm_ffn(h))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class STDCformer(nn.Module):
    def __init__(self, cfg: STDCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # V' = Conv^{f->d}(V) (1x1 conv == linear over channels)
        self.value_emb = nn.Sequential(nn.Linear(cfg.in_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.ste = STEmbedding(cfg)
        self.encoder = nn.ModuleList([STDCBlock(cfg) for _ in range(cfg.enc_layers)])
        self.cta = nn.ModuleList([CrossTimeAttention(cfg) for _ in range(cfg.cta_layers)])
        self.decoder = nn.ModuleList([STDCBlock(cfg) for _ in range(cfg.dec_layers)])
        self.final_norm = nn.LayerNorm(d)
        # Prediction head: Y_hat = Conv2d(H_future) in R^{T_f x n x f}
        self.head = nn.Conv2d(d, cfg.out_dim, kernel_size=1)

    def forward(self, batch: dict, return_gates: bool = False):
        """
        batch keys:
          x            (B, T_p, N, f)  normalized flows
          tod_x/tod_y  (B, T_p)/(B, T_f) long, time-of-day index
          dow_x/dow_y  optional long, day-of-week
          hol_x/hol_y  optional long, holiday flag
          exo_x/exo_y  optional float (B, T, t), weather etc.
          lap_pe       (N, lap_dim)    Laplacian eigenvectors
          spatial_feat optional (N, s)
        returns y_hat (B, T_f, N, out_dim) [, dict of gates]
        """
        lap = batch.get("lap_pe")
        sf = batch.get("spatial_feat")
        if not self.cfg.use_time_of_day and "tod_x" not in batch:
            B, dev = batch["x"].shape[0], batch["x"].device
            Tp, Tf = self.cfg.input_len, self.cfg.output_len
            batch = dict(batch,
                         tod_x=torch.arange(Tp, device=dev).expand(B, Tp),
                         tod_y=torch.arange(Tp, Tp + Tf, device=dev).expand(B, Tf))
        c_s, c_t_past = self.ste.confounders(lap, sf, batch["tod_x"], batch.get("dow_x"),
                                             batch.get("hol_x"), batch.get("exo_x"))
        _, c_t_fut = self.ste.confounders(lap, sf, batch["tod_y"], batch.get("dow_y"),
                                          batch.get("hol_y"), batch.get("exo_y"))
        ste_past = self.ste(c_s, c_t_past)                                    # (B,T_p,N,d)
        ste_fut = self.ste(c_s, c_t_fut)                                      # (B,T_f,N,d)

        gates = {"enc": [], "dec": []}
        # ---- E: encoder ----
        h = self.value_emb(batch["x"])
        for blk in self.encoder:
            h, g = blk(h, ste_past, c_s, c_t_past, return_gate=True)
            gates["enc"].append(g)
        # ---- M: cross-time mapping ----
        h_fut = None
        for cta in self.cta:
            h_fut = cta(h, ste_past, ste_fut, h_fut)
        # ---- D: decoder ----
        h = h_fut
        for blk in self.decoder:
            h, g = blk(h, ste_fut, c_s, c_t_fut, return_gate=True)
            gates["dec"].append(g)
        h = self.final_norm(h)
        y = self.head(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)             # (B,T_f,N,f)
        return (y, gates) if return_gates else y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
