"""MASO: DeepFP-style learned element-wise preconditioner unfolding (spec §6.3).

Per layer l and IRS n (after the deterministic block):
    g_n  = grad_matrix_free(...)                       (Euclidean gradient, B x M)
    r_n  = g_n - Re{g_n conj(phi_n)} phi_n               (Riemannian gradient)
    ghat = g_n / (||g_n|| + 1e-12)                      (norm over the FULL M)
    z_n  = [phi_n, ghat_n] padded with zeros / truncated to M_train each -> C^{2 M_train}
    lam  = Softplus(Re Psi_{l,n}(z_n))[:M]   (zeros for elements >= M_train)
    phi_n <- (phi_n + lam r_n) / |phi_n + lam r_n|
Elements beyond M_train are copied unchanged (exact), i.e. lambda = 0 there.

Psi_{l,n}: ComplexLinear(2Mt->2Mt) -> CReLU -> ComplexLinear(2Mt->2Mt) -> CReLU
           -> ComplexLinear(2Mt->Mt) -> real part -> Softplus.
Separate weights per (l, n): psi[l * N + n].

Parameters are float32 (real/imag parts kept separately); in forward they are cast
to the input's real dtype so complex128 evaluation works. Complex products use real
matmuls on (B, F) operands (avoids the CPU multithreaded batched complex matmul bug).
Init: nn.Linear's default (Kaiming-uniform, a=sqrt(5); bias U(+-1/sqrt(fan_in))) for each
of W_re, W_im, b_re, b_im, then scaled by 1/sqrt(2) so the complex weight/bias variance
matches the real nn.Linear default.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import fp_core as fp

from .common import UnfoldedNet


def complex_relu(z: torch.Tensor) -> torch.Tensor:
    """ReLU(Re z) + j ReLU(Im z) (DeepFP paper definition)."""
    return torch.complex(F.relu(z.real), F.relu(z.imag))


class ComplexReLU(nn.Module):
    def forward(self, z):
        return complex_relu(z)


class RealPart(nn.Module):
    def forward(self, z):
        return z.real


class ComplexLinear(nn.Module):
    """y = W x + b with complex W, b stored as real float32 parameters (W_re, W_im, b_re, b_im)."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.in_f, self.out_f = int(in_f), int(out_f)
        self.W_re = nn.Parameter(torch.empty(out_f, in_f))
        self.W_im = nn.Parameter(torch.empty(out_f, in_f))
        self.b_re = nn.Parameter(torch.empty(out_f))
        self.b_im = nn.Parameter(torch.empty(out_f))
        self.reset_parameters()

    def reset_parameters(self):
        s = 1.0 / math.sqrt(2.0)
        bound = 1.0 / math.sqrt(self.in_f)
        with torch.no_grad():
            for w in (self.W_re, self.W_im):
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                w.mul_(s)
            for b in (self.b_re, self.b_im):
                nn.init.uniform_(b, -bound, bound)
                b.mul_(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_f) complex
        rd = x.real.dtype
        Wr, Wi = self.W_re.to(rd), self.W_im.to(rd)
        br, bi = self.b_re.to(rd), self.b_im.to(rd)
        xr, xi = x.real, x.imag
        out_re = xr @ Wr.T - xi @ Wi.T + br
        out_im = xr @ Wi.T + xi @ Wr.T + bi
        return torch.complex(out_re, out_im)


def _mlp(Mt: int) -> nn.Sequential:
    return nn.Sequential(
        ComplexLinear(2 * Mt, 2 * Mt), ComplexReLU(),
        ComplexLinear(2 * Mt, 2 * Mt), ComplexReLU(),
        ComplexLinear(2 * Mt, Mt), RealPart(), nn.Softplus(),
    )


def _fit(x: torch.Tensor, Mt: int) -> torch.Tensor:
    """Zero-pad (M < Mt) or truncate (M > Mt) the last dim to Mt."""
    M = x.shape[-1]
    if M < Mt:
        return F.pad(x, (0, Mt - M))
    return x[..., :Mt]


class MASO(UnfoldedNet):
    def __init__(self, L: int = 10, N: int = 2, M_train: int = 80):
        super().__init__(L)
        self.N, self.M_train = int(N), int(M_train)
        self.psi = nn.ModuleList([_mlp(self.M_train) for _ in range(self.L * self.N)])

    def step_sizes(self, l: int, phi: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """lambda (B,N,M) >= 0 for layer l; zeros beyond M_train."""
        M, Mt = phi.shape[-1], self.M_train
        ghat = g / (torch.linalg.vector_norm(g, dim=-1, keepdim=True) + 1e-12)
        z = torch.cat([_fit(phi, Mt), _fit(ghat, Mt)], dim=-1)  # (B,N,2Mt)
        lam = torch.stack([self.psi[l * self.N + n](z[:, n]) for n in range(self.N)], dim=1)
        if M <= Mt:
            return lam[..., :M]
        return F.pad(lam, (0, M - Mt))

    def phase_update(self, l, ch, p_new, phi, aux):
        g = fp.grad_matrix_free(ch, p_new, phi, aux)
        r = g - (g * phi.conj()).real * phi
        lam = self.step_sizes(l, phi, g).to(phi.real.dtype)
        u = phi + lam * r
        out = u / u.abs()
        M, Mt = phi.shape[-1], self.M_train
        if M > Mt:  # lambda = 0 on the tail: keep those phases exactly
            out = torch.cat([out[..., :Mt], phi[..., Mt:]], dim=-1)
        return out
