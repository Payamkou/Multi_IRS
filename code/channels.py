"""Multi-IRS uplink channel model (units: sigma^2 = 1 internally, P in W)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from config import Config

F64, C128 = torch.float64, torch.complex128


@dataclass
class ChannelBatch:
    hd: torch.Tensor   # (B,K) complex, scaled by 1/sigma
    b: torch.Tensor    # (B,N,K,M) complex, diag(g_n) conj(h_r), g scaled by 1/sigma
    x: torch.Tensor    # (B,N,K) real {0,1}
    w: torch.Tensor    # (B,K)
    P: torch.Tensor    # (B,K) W
    sigma2: float = 1.0

    @property
    def B(self): return self.b.shape[0]
    @property
    def N(self): return self.b.shape[1]
    @property
    def K(self): return self.b.shape[2]
    @property
    def M(self): return self.b.shape[3]

    def to(self, device, cdtype=C128) -> "ChannelBatch":
        rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
        return ChannelBatch(self.hd.to(device=device, dtype=cdtype), self.b.to(device=device, dtype=cdtype),
                            self.x.to(device=device, dtype=rdtype), self.w.to(device=device, dtype=rdtype),
                            self.P.to(device=device, dtype=rdtype), self.sigma2)

    def slice(self, idx) -> "ChannelBatch":
        return ChannelBatch(self.hd[idx], self.b[idx], self.x[idx], self.w[idx], self.P[idx], self.sigma2)


def path_loss_db(d_m: torch.Tensor) -> torch.Tensor:
    return 128.1 + 37.6 * torch.log10(d_m / 1000.0)


def ula_steering(sin_psi: torch.Tensor, M: int) -> torch.Tensor:
    m = torch.arange(M, dtype=sin_psi.dtype, device=sin_psi.device)
    return torch.exp(1j * math.pi * sin_psi.unsqueeze(-1) * m)


def associate(b: torch.Tensor) -> torch.Tensor:
    """One-hot (B,N,K) of argmax_n ||b_{k,n}||."""
    nrm = b.norm(dim=-1)  # (B,N,K)
    idx = nrm.argmax(dim=1, keepdim=True)
    return torch.zeros_like(nrm).scatter_(1, idx, 1.0)


def sample_users(cfg: Config, B: int, K: int, gen: torch.Generator) -> torch.Tensor:
    """(B,K,2) positions uniform in the disk."""
    r = cfg.disk_radius * torch.sqrt(torch.rand(B, K, generator=gen, dtype=F64))
    ang = 2 * math.pi * torch.rand(B, K, generator=gen, dtype=F64)
    c = torch.tensor(cfg.disk_center, dtype=F64)
    return torch.stack([c[0] + r * torch.cos(ang), c[1] + r * torch.sin(ang)], dim=-1)


def _cn(gen, *shape):
    return torch.complex(torch.randn(*shape, generator=gen, dtype=F64),
                         torch.randn(*shape, generator=gen, dtype=F64)) / math.sqrt(2.0)


def _amp(pl_db, shadow):
    return torch.sqrt(10.0 ** (-(pl_db + shadow) / 10.0))


def generate_channels(cfg: Config, B: int, seed: int, M=None, K=None, crn_dbm=None):
    M = cfg.M if M is None else int(M)
    K = cfg.K if K is None else int(K)
    crn_dbm = cfg.crn_dbm if crn_dbm is None else float(crn_dbm)
    N = len(cfg.irs_pos)
    assert cfg.N == N, "cfg.N must equal len(cfg.irs_pos)"
    gen = torch.Generator().manual_seed(int(seed))
    gph = torch.Generator().manual_seed(int(seed) + 7)
    inv_sigma = 1.0 / math.sqrt(10.0 ** ((cfg.noise_dbm - 30.0) / 10.0))
    kap = cfg.kappa
    los_w, nlos_w = math.sqrt(kap / (1 + kap)), math.sqrt(1 / (1 + kap))

    users = sample_users(cfg, B, K, gen)                                   # (B,K,2)
    bs = torch.tensor(cfg.bs, dtype=F64)
    irs = torch.tensor(cfg.irs_pos, dtype=F64)                              # (N,2)

    d_ub = (users - bs).norm(dim=-1)                                        # (B,K)
    d_ui = (users.unsqueeze(1) - irs.view(1, N, 1, 2)).norm(dim=-1)         # (B,N,K)
    d_ib = (irs - bs).norm(dim=-1)                                          # (N,)

    sh = cfg.shadow_db
    s_d = sh * torch.randn(B, K, generator=gen, dtype=F64)
    s_ui = sh * torch.randn(B, N, K, generator=gen, dtype=F64)
    s_ib = sh * torch.randn(B, N, generator=gen, dtype=F64)

    # direct link (Rayleigh, NLOS penalty)
    hd = _amp(path_loss_db(d_ub) + cfg.nlos_db, s_d) * _cn(gen, B, K)

    # user-IRS: LoS steering at the IRS toward the user, sin(psi) = dy/d
    sin_u = (users[..., 1].unsqueeze(1) - irs[:, 1].view(1, N, 1)) / d_ui   # (B,N,K)
    a_u = ula_steering(sin_u, M)                                            # (B,N,K,M)
    hr = _amp(path_loss_db(d_ui), s_ui).unsqueeze(-1) * (
        los_w * a_u + nlos_w * _cn(gen, B, N, K, M))

    # IRS-BS: LoS steering at the IRS toward the BS (fixed by geometry)
    sin_b = (bs[1] - irs[:, 1]) / d_ib                                      # (N,)
    a_b = ula_steering(sin_b, M)                                            # (N,M)
    g = _amp(path_loss_db(d_ib).unsqueeze(0), s_ib).unsqueeze(-1) * (
        los_w * a_b.unsqueeze(0) + nlos_w * _cn(gen, B, N, M))              # (B,N,M)

    hd = hd * inv_sigma
    g = g * inv_sigma
    b = g.unsqueeze(2) * hr.conj()                                          # (B,N,K,M)

    even = (torch.arange(K) % 2 == 0)
    cyc = torch.arange(K) // 2
    crn_w = torch.tensor(cfg.crn_weights, dtype=F64)[cyc % len(cfg.crn_weights)]
    pmn_w = torch.tensor(cfg.pmn_weights, dtype=F64)[cyc % len(cfg.pmn_weights)]
    w1 = torch.where(even, crn_w, pmn_w)
    p_crn = 10.0 ** ((crn_dbm - 30.0) / 10.0)
    p_pmn = 10.0 ** ((crn_dbm + cfg.pmn_offset_db - 30.0) / 10.0)
    P1 = torch.where(even, torch.tensor(p_crn, dtype=F64), torch.tensor(p_pmn, dtype=F64))
    w = w1.expand(B, K).clone()
    P = P1.expand(B, K).clone()

    x = associate(b)
    theta0 = 2 * math.pi * torch.rand(B, N, M, generator=gph, dtype=F64)
    return ChannelBatch(hd=hd.to(C128), b=b.to(C128), x=x, w=w, P=P, sigma2=1.0), theta0
