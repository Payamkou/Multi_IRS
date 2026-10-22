"""Training (spec §8).

Loss section
------------
* ``loss_supervised``      Stage I phase alignment, mean_{B,N,M} (1 - Re{phi^(L) conj(phi*)}).
* ``loss_unsup_layer``     Stage II composite mean_B[-a R_nats - (1-a) f3] at one layer's output.
  Inside f3, wt = w(1 + alpha_in) uses the alpha the layer actually used (its INPUT-state
  alpha), DETACHED; eps is recomputed in closed form from the output (p_l, phi_l) with
  gradients flowing through it. Both terms are in nats.
* ``loss_deep_supervision`` sum_{l=1..L} rho_l loss_unsup_layer(layer-l output), rho_l = l / sum_j j.
"""
from __future__ import annotations

import torch

import fp_core as fp
from channels import ChannelBatch
from models.common import Trajectory


# ---------------------------------------------------------------- losses
def loss_supervised(phi_L: torch.Tensor, phi_star: torch.Tensor) -> torch.Tensor:
    """mean over (B,N,M) of 1 - Re{phi_L conj(phi_star)}."""
    return (1.0 - (phi_L * phi_star.conj()).real).mean()


def loss_unsup_layer(ch: ChannelBatch, p_l: torch.Tensor, phi_l: torch.Tensor,
                     alpha_in_l: torch.Tensor, a: float = 0.5) -> torch.Tensor:
    """mean_B[-a R_nats - (1-a) f3] with detached input-state alpha and closed-form eps."""
    wt = ch.w * (1 + alpha_in_l.detach())
    H = fp.effective_channel(ch, phi_l)
    D = fp.received_power(ch, p_l, H)
    eps = fp.safe_sqrt(wt * p_l) * H / D[:, None]          # grads flow through p_l, phi_l
    f3 = fp.surrogate_f3(ch, p_l, phi_l, wt, eps)
    R = fp.wsr(ch, p_l, phi_l, base="e")
    return (-a * R - (1 - a) * f3).mean()


def rho_weights(L: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """rho_l = l / sum_{j=1..L} j, l = 1..L  -> (L,)."""
    l = torch.arange(1, L + 1, dtype=dtype, device=device)
    return l / l.sum()


def loss_deep_supervision(ch: ChannelBatch, traj: Trajectory, a: float = 0.5) -> torch.Tensor:
    """sum_{l=1..L} rho_l loss_unsup_layer(ch, traj.p[l], traj.phi[l], traj.alpha_in[l-1], a)."""
    L = traj.alpha_in.shape[0]
    rho = rho_weights(L, dtype=traj.p.dtype, device=traj.p.device)
    total = traj.p.new_zeros(())
    for l in range(1, L + 1):  # loop over layers (allowed)
        total = total + rho[l - 1] * loss_unsup_layer(ch, traj.p[l], traj.phi[l], traj.alpha_in[l - 1], a)
    return total
