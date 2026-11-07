import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ["full", "fac_st", "fac_ts", "TS", "ST", "TST", "STS", "TSST", "STTS"]


class DropPath(nn.Module):
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

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, L, D)))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(self.drop(F.silu(a) * b)))


class GTB(nn.Module):
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


class SpatialGTB(GTB):
    def forward(self, x):
        B, T, N, D = x.shape
        return super().forward(x.reshape(B * T, N, D)).view(B, T, N, D)


class TemporalGTB(GTB):
    def forward(self, x):
        B, T, N, D = x.shape
        y = x.transpose(1, 2).reshape(B * N, T, D)
        return super().forward(y).view(B, N, T, D).transpose(1, 2)


class FullGTB(GTB):
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


def block_order(variant, num_blocks):
    if variant == "full":
        return "F" * num_blocks
    if variant in ("fac_st", "fac_ts"):
        assert num_blocks % 2 == 0, "factorized variants need an even num_blocks"
        half = num_blocks // 2
        return "S" * half + "T" * half if variant == "fac_st" else "T" * half + "S" * half
    assert num_blocks % len(variant) == 0, f"num_blocks={num_blocks} is not divisible by len('{variant}')"
    return variant * (num_blocks // len(variant))


class PredFormer(nn.Module):
    def __init__(self, in_shape, patch_size, dim=256, heads=8, mlp_dim=1024, num_blocks=24,
                 variant="TSST", attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        T, C, H, W = in_shape
        p = patch_size
        assert H % p == 0 and W % p == 0, f"H and W ({H}x{W}) must be divisible by patch_size {p}"
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.T, self.C, self.H, self.W, self.p = T, C, H, W, p
        self.Hp, self.Wp = H // p, W // p
        self.N = self.Hp * self.Wp

        self.patch_embed = nn.Linear(p * p * C, dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.register_buffer("pos_embed", sinusoidal_table(T * self.N, dim).view(1, T, self.N, dim),
                             persistent=False)
        block = {"S": SpatialGTB, "T": TemporalGTB, "F": FullGTB}
        self.blocks = nn.ModuleList(block[c](dim, heads, mlp_dim, attn_drop, drop, drop_path)
                                    for c in block_order(variant, num_blocks))
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

    def patchify(self, x):
        B, p = x.shape[0], self.p
        x = x.reshape(B, self.T, self.C, self.Hp, p, self.Wp, p).permute(0, 1, 3, 5, 4, 6, 2)
        return x.reshape(B, self.T, self.N, p * p * self.C)

    def unpatchify(self, x):
        B, p = x.shape[0], self.p
        x = x.reshape(B, self.T, self.Hp, self.Wp, p, p, self.C).permute(0, 1, 6, 2, 4, 3, 5)
        return x.reshape(B, self.T, self.C, self.H, self.W)

    def forward(self, x):
        x = self.patch_norm(self.patch_embed(self.patchify(x))) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.unpatchify(self.head(self.norm(x)))


class Horizon(nn.Module):
    def __init__(self, model, t_in, t_out):
        super().__init__()
        self.model, self.t_in, self.t_out = model, t_in, t_out

    def forward(self, x):
        outs, n, cur = [], 0, x
        while n < self.t_out:
            cur = self.model(cur)
            outs.append(cur)
            n += cur.shape[1]
        return torch.cat(outs, dim=1)[:, :self.t_out]


def with_horizon(model, t_in, t_out):
    return model if t_in == t_out else Horizon(model, t_in, t_out)
