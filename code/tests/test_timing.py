"""Spec §7.2: device/precision-fair timing."""
import dataclasses

import numpy as np
import pytest
import torch

import ao
import timing
from channels import generate_channels
from config import Config
from models import build_model

C128, C64 = torch.complex128, torch.complex64


def _one(M=16, K=4, B=3):
    cfg = dataclasses.replace(Config(), M=M, K=K)
    ch, theta0 = generate_channels(cfg, B, seed=5, M=M, K=K)
    return cfg, ch, theta0


def test_times_positive_and_shape():
    cfg, ch, theta0 = _one()
    ch1, th1 = ch.slice([0]).to("cpu", C128), theta0[[0]]
    fonr = build_model("fonr", cfg, mu_init=0.2).to_precision(C128)
    for st in (fonr, ao.AOStepper("efficient")):
        t = timing.time_steps(st, ch1, th1, 5, "cpu", C128, warmup=1)
        assert t.shape == (5,) and np.all(t > 0) and np.all(np.isfinite(t))
    assert timing.set_timing_threads(2) == 2


def test_same_device_and_dtype_enforced():
    cfg, ch, theta0 = _one()
    ch1, th1 = ch.slice([0]).to("cpu", C128), theta0[[0]]
    fonr = build_model("fonr", cfg, mu_init=0.2)  # float32 params
    with pytest.raises(ValueError):
        timing.time_steps(fonr, ch1, th1, 2, "cpu", C128, warmup=0)
    fonr.to_precision(C128)
    timing.time_steps(fonr, ch1, th1, 2, "cpu", C128, warmup=0)
    maso = build_model("maso", cfg, step_scale=1.0)
    assert maso.to_precision(C128) is maso
    assert all(b.dtype == torch.float64 for b in maso.buffers())
    assert all(p.dtype == torch.float64 for p in maso.parameters())
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            timing.time_steps(fonr, ch1, th1, 2, "cuda", C128, warmup=0)


def test_until_converged_and_per_iteration():
    cfg, ch, theta0 = _one()
    ch1, th1 = ch.slice([0]).to("cpu", C128), theta0[[0]]
    st = ao.AOStepper("efficient")
    times, wsr = timing.time_until_converged(st, ch1, th1, 1e-5, 50, "cpu", C128)
    assert len(wsr) == len(times) + 1 and 1 <= len(times) <= 50
    assert wsr[-1] >= wsr[0]
    pit = timing.per_iteration_time(st, dict(ch=ch, theta0=theta0), [0, 2], "cpu", C128, n_steps=3)
    assert pit.shape == (2,) and np.all(pit > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_path_syncs():
    cfg, ch, theta0 = _one()
    ch1, th1 = ch.slice([0]).to("cuda", C64), theta0[[0]]
    fonr = build_model("fonr", cfg, mu_init=0.2).to_precision(C64).to("cuda")
    t = timing.time_steps(fonr, ch1, th1, 3, "cuda", C64, warmup=1)
    assert np.all(np.isfinite(t)) and np.all(t > 0)
