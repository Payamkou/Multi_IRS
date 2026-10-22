"""Shared deep-unfolding base (spec §6.5).

Every layer l: ``p_new, aux = deterministic_block(ch, p, phi)`` (parameter-free power
layer and auxiliary variables, aux built from the layer's INPUT state), then the
model-specific ``phase_update(l, ch, p_new, phi, aux)``. Same stepper API as
``ao.AOStepper`` so timing code treats AO and nets identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import fp_core as fp
from channels import ChannelBatch


@dataclass
class Trajectory:
    p: torch.Tensor         # (L+1,B,K); index 0 = initial state
    phi: torch.Tensor       # (L+1,B,N,M)
    alpha_in: torch.Tensor  # (L,B,K) alpha (SINR) of each layer's input state


class UnfoldedNet(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.L = int(L)

    def phase_update(self, l: int, ch: ChannelBatch, p_new: torch.Tensor, phi: torch.Tensor,
                     aux: fp.Aux) -> torch.Tensor:
        raise NotImplementedError

    def _layer(self, l: int, ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor):
        p_new, aux = fp.deterministic_block(ch, p, phi)
        return p_new, self.phase_update(l, ch, p_new, phi, aux), aux

    def forward(self, ch: ChannelBatch, p0: torch.Tensor, phi0: torch.Tensor) -> Trajectory:
        ps, phis, alphas = [p0], [phi0], []
        p, phi = p0, phi0
        for l in range(self.L):
            p, phi, aux = self._layer(l, ch, p, phi)
            ps.append(p)
            phis.append(phi)
            alphas.append(aux.alpha)
        return Trajectory(p=torch.stack(ps), phi=torch.stack(phis), alpha_in=torch.stack(alphas))

    # ---- stepper API (same as ao.AOStepper)
    def init(self, ch: ChannelBatch, p0: torch.Tensor, phi0: torch.Tensor):
        return (p0, phi0)

    def step(self, i: int, ch: ChannelBatch, state):
        p, phi = state
        p_new, phi_new, _ = self._layer(i, ch, p, phi)
        return (p_new, phi_new)
