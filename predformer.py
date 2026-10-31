"""
PredFormer in PyTorch, written from the paper (Tang et al., "Video Prediction
Transformers without Recurrence or Convolution", arXiv:2410.04733). No
pretrained weights; everything trains from scratch.
 
Pipeline (paper Sec. 3.1):
    frames -> non-overlapping p x p patches -> Linear(D) -> LayerNorm
           -> + fixed sinusoidal spatiotemporal position encoding
           -> PredFormer encoder (Gated Transformer Blocks in some arrangement)
           -> Linear back to p*p*C -> un-patchify -> predicted frames
 
Tensors are (B, T, C, H, W). As in the paper, T_out == T_in.
"""
 
import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample."""
 
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
 
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep
 
 
class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
 
    def forward(self, x):                                    # (B, L, D)
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, L, D)))
 
 
class SwiGLU(nn.Module):
    """W3( SiLU(x W1) * (x W2) ), the FFN in a Gated Transformer Block."""
 
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(self.drop(F.silu(a) * b)))
 
 
class GTB(nn.Module):
    """Gated Transformer Block (paper Eq. 3), pre-norm:
        Y  = MSA(LN(Z)) + Z
        Z' = SwiGLU(LN(Y)) + Y
    """
 
    def __init__(self, dim, heads, mlp_dim, attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, mlp_dim, drop)
        self.drop_path = DropPath(drop_path)
 
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.ffn(self.norm2(x)))
 
 
# ---------------------------------------------------------------------------
# Axis wrappers: each takes and returns (B, T, N, D)
# ---------------------------------------------------------------------------
class SpatialGTB(GTB):
    """Attention over the N patches within each frame."""
 
    def forward(self, x):
        B, T, N, D = x.shape
        return super().forward(x.reshape(B * T, N, D)).view(B, T, N, D)
 
 
class TemporalGTB(GTB):
    """Attention over the T time steps at each patch location."""
 
    def forward(self, x):
        B, T, N, D = x.shape
        y = x.transpose(1, 2).reshape(B * N, T, D)
        return super().forward(y).view(B, N, T, D).transpose(1, 2)
 
 
class FullGTB(GTB):
    """Joint attention over all T*N tokens."""
 
    def forward(self, x):
        B, T, N, D = x.shape
        return super().forward(x.reshape(B, T * N, D)).view(B, T, N, D)
 
 
def sinusoidal_table(n, dim):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(dim, dtype=torch.float32)[None, :]
    angle = pos / torch.pow(10000.0, (2 * torch.div(i, 2, rounding_mode="floor")) / dim)
    table = torch.zeros(n, dim)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table
 
 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
VARIANTS = ["full", "fac_st", "fac_ts", "TS", "ST", "TST", "STS", "TSST", "STTS"]
 
 
class PredFormer(nn.Module):
    """
    Args:
        in_shape:   (T, C, H, W) of the input clip.
        patch_size: p; H and W must be divisible by it.
        dim, heads, mlp_dim: GTB width, heads, SwiGLU hidden size.
        num_blocks: total GTBs. Interleaved variants repeat their pattern
                    num_blocks / len(pattern) times (24 blocks + "TSST" =
                    6 PredFormer layers, the paper's Moving MNIST setup).
        variant:    "full", "fac_st", "fac_ts" (factorized), or an interleaved
                    pattern: "TS", "ST", "TST", "STS", "TSST", "STTS".
                    The paper recommends "TSST" as the default.
        attn_drop, drop, drop_path: regularization. Drop path is uniform across
                    depth, which the paper found beats the usual linear ramp.
    """
 
    def __init__(self, in_shape, patch_size, dim=256, heads=8, mlp_dim=1024,
                 num_blocks=24, variant="TSST", attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        T, C, H, W = in_shape
        p = patch_size
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.T, self.C, self.H, self.W, self.p = T, C, H, W, p
        self.Hp, self.Wp = H // p, W // p
        self.N = self.Hp * self.Wp
 
        # Patch embedding
        self.patch_embed = nn.Linear(p * p * C, dim)
        self.patch_norm = nn.LayerNorm(dim)
 
        # Fixed absolute sinusoidal PE over the T x N token grid
        self.register_buffer(
            "pos_embed", sinusoidal_table(T * self.N, dim).view(1, T, self.N, dim),
            persistent=False)
 
        # Encoder
        if variant == "full":
            order = "F" * num_blocks
        elif variant in ("fac_st", "fac_ts"):
            assert num_blocks % 2 == 0, "factorized variants need even num_blocks"
            h = num_blocks // 2
            order = "S" * h + "T" * h if variant == "fac_st" else "T" * h + "S" * h
        else:
            assert num_blocks % len(variant) == 0, (
                f"num_blocks={num_blocks} is not divisible by len('{variant}')")
            order = variant * (num_blocks // len(variant))
        cls = {"S": SpatialGTB, "T": TemporalGTB, "F": FullGTB}
        self.blocks = nn.ModuleList(
            cls[c](dim, heads, mlp_dim, attn_drop, drop, drop_path) for c in order)
 
        # Patch recovery: linear decoder
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, p * p * C)
 
        self.apply(self._init_weights)
 
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
 
    def patchify(self, x):                     # (B,T,C,H,W) -> (B,T,N,p*p*C)
        B, p = x.shape[0], self.p
        x = x.reshape(B, self.T, self.C, self.Hp, p, self.Wp, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2)     # B,T,Hp,Wp,p,p,C
        return x.reshape(B, self.T, self.N, p * p * self.C)
 
    def unpatchify(self, x):                   # (B,T,N,p*p*C) -> (B,T,C,H,W)
        B, p = x.shape[0], self.p
        x = x.reshape(B, self.T, self.Hp, self.Wp, p, p, self.C)
        x = x.permute(0, 1, 6, 2, 4, 3, 5)     # B,T,C,Hp,p,Wp,p
        return x.reshape(B, self.T, self.C, self.H, self.W)
 
    def forward(self, x):
        x = self.patch_norm(self.patch_embed(self.patchify(x)))
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.unpatchify(self.head(self.norm(x)))
 
 
# ---------------------------------------------------------------------------
# Paper hyperparameters (Table 9)
# ---------------------------------------------------------------------------
CONFIGS = {
    "mmnist": dict(
        model=dict(in_shape=(10, 1, 64, 64), patch_size=8, dim=256, heads=8,
                   mlp_dim=1024, num_blocks=24, variant="TSST",
                   attn_drop=0.0, drop=0.0, drop_path=0.0),
        train=dict(batch_size=16, lr=1e-3, weight_decay=1e-2, epochs=2000),
    ),
    "taxibj": dict(
        # paper: 8 GTBs for most variants, 6 for the triplet ones (TST / STS)
        model=dict(in_shape=(4, 2, 32, 32), patch_size=4, dim=256, heads=8,
                   mlp_dim=1024, num_blocks=8, variant="TSST",
                   attn_drop=0.1, drop=0.1, drop_path=0.1),
        train=dict(batch_size=16, lr=1e-3, weight_decay=1e-2, epochs=200),
    ),
}
