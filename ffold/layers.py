"""F-Fold primitive layers (AlphaFold3 / OpenFold3 shapes).

Implements the building blocks the AF3 supplement specifies: SwiGLU
transitions, triangular multiplicative update, triangular attention,
attention with pair bias, and adaptive layer norm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lecun_normal_(w: torch.Tensor) -> None:
    fan_in = w.shape[-1]
    nn.init.trunc_normal_(w, std=math.sqrt(1.0 / fan_in), a=-2.0, b=2.0)


class Linear(nn.Linear):
    """Linear with the init policies AF3 distinguishes between."""

    def __init__(self, d_in: int, d_out: int, bias: bool = True,
                 init: str = "default"):
        super().__init__(d_in, d_out, bias=bias)
        with torch.no_grad():
            if init == "default":
                _lecun_normal_(self.weight)
                if bias:
                    self.bias.fill_(0.0)
            elif init == "final":
                self.weight.fill_(0.0)
                if bias:
                    self.bias.fill_(0.0)
            elif init == "gating":
                self.weight.fill_(0.0)
                if bias:
                    self.bias.fill_(1.0)
            elif init == "gating_ada_zero":
                self.weight.fill_(0.0)
                if bias:
                    self.bias.fill_(-2.0)
            elif init == "relu":
                nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
                if bias:
                    self.bias.fill_(0.0)
            else:
                raise ValueError(f"unknown init {init!r}")


class LayerNorm(nn.Module):
    def __init__(self, c: int, create_scale: bool = True,
                 create_offset: bool = True):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.ones(c)) if create_scale else None
        self.bias = nn.Parameter(torch.zeros(c)) if create_offset else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.c,), self.weight, self.bias, 1e-5)


class AdaLayerNorm(nn.Module):
    """Conditioned layer norm: scale/shift produced from a conditioning vector."""

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.ln_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.ln_s = LayerNorm(c_s, create_offset=False)
        self.linear_g = Linear(c_s, c_a, bias=True, init="final")
        self.linear_s = Linear(c_s, c_a, bias=False, init="final")

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s = self.ln_s(s)
        return torch.sigmoid(self.linear_g(s)) * self.ln_a(a) + self.linear_s(s)


class Transition(nn.Module):
    """SwiGLU transition (AF3 Alg. 11)."""

    def __init__(self, c: int, n: int):
        super().__init__()
        self.ln = LayerNorm(c)
        self.linear_ab = Linear(c, 2 * n * c, bias=False)
        self.linear_out = Linear(n * c, c, bias=False, init="final")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear_ab(self.ln(x)).chunk(2, dim=-1)
        return self.linear_out(F.silu(a) * b)


class ConditionedTransition(nn.Module):
    """SwiGLU transition with AdaLN conditioning and output gate (AF3 Alg. 25)."""

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.ada_ln = AdaLayerNorm(c_a, c_s)
        self.linear_ab = Linear(c_a, 2 * n * c_a, bias=False)
        self.linear_out = Linear(n * c_a, c_a, bias=False)
        self.linear_gate = Linear(c_s, c_a, bias=True, init="gating_ada_zero")

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.ada_ln(a, s)
        x, g = self.linear_ab(h).chunk(2, dim=-1)
        out = self.linear_out(F.silu(x) * g)
        return torch.sigmoid(self.linear_gate(s)) * out


class TriangleMultiplication(nn.Module):
    def __init__(self, c_z: int, c_hidden: int, outgoing: bool = True):
        super().__init__()
        self.outgoing = outgoing
        self.ln_in = LayerNorm(c_z)
        self.linear_a_p = Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = Linear(c_z, c_hidden, bias=False, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = Linear(c_z, c_hidden, bias=False, init="gating")
        self.ln_out = LayerNorm(c_hidden)
        self.linear_z = Linear(c_z, c_z, bias=False, init="gating")
        self.linear_out = Linear(c_hidden, c_z, bias=False, init="final")

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None):
        z = self.ln_in(z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        if mask is not None:
            m = mask[..., None]
            a, b = a * m, b * m
        if self.outgoing:
            x = torch.einsum("...ikc,...jkc->...ijc", a, b)
        else:
            x = torch.einsum("...kic,...kjc->...ijc", a, b)
        g = torch.sigmoid(self.linear_z(z))
        return g * self.linear_out(self.ln_out(x))


class TriangleAttention(nn.Module):
    def __init__(self, c_z: int, c_hidden: int, no_heads: int,
                 starting: bool = True):
        super().__init__()
        self.starting = starting
        self.h, self.c = no_heads, c_hidden
        self.ln = LayerNorm(c_z)
        self.linear_q = Linear(c_z, no_heads * c_hidden, bias=False)
        self.linear_k = Linear(c_z, no_heads * c_hidden, bias=False)
        self.linear_v = Linear(c_z, no_heads * c_hidden, bias=False)
        self.linear_b = Linear(c_z, no_heads, bias=False)
        self.linear_g = Linear(c_z, no_heads * c_hidden, bias=False,
                               init="gating")
        self.linear_o = Linear(no_heads * c_hidden, c_z, bias=False,
                               init="final")

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None):
        # `mask` is a 1-D token mask [..., L]; transposing z turns
        # starting-node attention into ending-node attention.
        if not self.starting:
            z = z.transpose(-3, -2)
        zn = self.ln(z)
        *B, I, J, _ = zn.shape
        q = self.linear_q(zn).view(*B, I, J, self.h, self.c)
        k = self.linear_k(zn).view(*B, I, J, self.h, self.c)
        v = self.linear_v(zn).view(*B, I, J, self.h, self.c)
        # logits[..., h, i, j, k] = q[i,j]·k[i,k]/sqrt(c) + b[j,k]
        logits = torch.einsum("...ijhc,...ikhc->...hijk", q, k) / math.sqrt(self.c)
        b_jk = self.linear_b(zn)                                  # ...IJh
        b_jk = b_jk.movedim(-1, -3).unsqueeze(-3)                 # ...h,1,J,K
        logits = logits + b_jk
        if mask is not None:
            logits = logits + (1.0 - mask[..., None, None, None, :]) * -1e9
        att = torch.softmax(logits, dim=-1)
        out = torch.einsum("...hijk,...ikhc->...ijhc", att, v)
        g = torch.sigmoid(self.linear_g(zn)).view(*B, I, J, self.h, self.c)
        out = (out * g).reshape(*B, I, J, self.h * self.c)
        out = self.linear_o(out)
        if not self.starting:
            out = out.transpose(-3, -2)
        return out


class AttentionPairBias(nn.Module):
    """Single-representation attention biased by the pair representation."""

    def __init__(self, c_a: int, c_z: int, c_hidden: int, no_heads: int,
                 c_s: int | None = None, use_ada_layer_norm: bool = False):
        super().__init__()
        self.h, self.c = no_heads, c_hidden
        self.use_ada = use_ada_layer_norm
        if use_ada_layer_norm:
            assert c_s is not None
            self.ada_ln = AdaLayerNorm(c_a, c_s)
            self.linear_ada_out = Linear(c_s, c_a, bias=True,
                                         init="gating_ada_zero")
        else:
            self.ln_a = LayerNorm(c_a)
        self.linear_q = Linear(c_a, no_heads * c_hidden, bias=True)
        self.linear_k = Linear(c_a, no_heads * c_hidden, bias=False)
        self.linear_v = Linear(c_a, no_heads * c_hidden, bias=False)
        self.ln_z = LayerNorm(c_z, create_offset=False)
        self.linear_z = Linear(c_z, no_heads, bias=False)
        self.linear_g = Linear(c_a, no_heads * c_hidden, bias=False,
                               init="gating")
        self.linear_o = Linear(no_heads * c_hidden, c_a, bias=False,
                               init="final")

    def forward(self, a: torch.Tensor, z: torch.Tensor,
                s: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ada_ln(a, s) if self.use_ada else self.ln_a(a)
        *B, N, _ = h.shape
        q = self.linear_q(h).view(*B, N, self.h, self.c)
        k = self.linear_k(h).view(*B, N, self.h, self.c)
        v = self.linear_v(h).view(*B, N, self.h, self.c)
        bias = self.linear_z(self.ln_z(z))                 # ...NNh
        logits = torch.einsum("...ihc,...jhc->...hij", q, k) / math.sqrt(self.c)
        logits = logits + bias.movedim(-1, -3)             # ...hNN
        if mask is not None:
            logits = logits + (1.0 - mask[..., None, None, :]) * -1e9
        att = torch.softmax(logits, dim=-1)
        out = torch.einsum("...hij,...jhc->...ihc", att, v)
        g = torch.sigmoid(self.linear_g(h)).view(*B, N, self.h, self.c)
        out = self.linear_o((out * g).reshape(*B, N, self.h * self.c))
        if self.use_ada:
            out = torch.sigmoid(self.linear_ada_out(s)) * out
        return out
