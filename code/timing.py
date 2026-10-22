"""Device/precision-fair per-step latency measurement (spec §7.2).

Works for ``ao.AOStepper`` and ``models.common.UnfoldedNet`` alike (same init/step API).
Models must already sit on the requested device with the matching real dtype
(``model.to_precision(cdtype).to(device)``); otherwise ValueError is raised.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import ao
import fp_core as fp


def set_timing_threads(n: int) -> int:
    torch.set_num_threads(int(n))
    return torch.get_num_threads()


def _real_dtype(cdtype):
    return torch.float32 if cdtype == torch.complex64 else torch.float64


def _check_stepper(stepper, device, cdtype) -> None:
    if not isinstance(stepper, torch.nn.Module):
        return
    rdtype = _real_dtype(cdtype)
    dev = torch.device(device)
    for name, t in list(stepper.named_parameters()) + list(stepper.named_buffers()):
        if t.device.type != dev.type or (dev.index is not None and t.device.index != dev.index):
            raise ValueError(f"{name} is on {t.device}, expected {dev}")
        if t.is_floating_point() and t.dtype != rdtype:
            raise ValueError(f"{name} has dtype {t.dtype}, expected {rdtype}")


def _prepare(stepper, ch1, theta0_1, device, cdtype):
    _check_stepper(stepper, device, cdtype)
    ch1 = ch1.to(device, cdtype)
    p0, phi0 = ao.init_state(ch1, theta0_1.to(device))
    return ch1, stepper.init(ch1, p0, phi0)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _timed_step(stepper, i, ch1, state, device):
    _sync(device)
    t0 = time.perf_counter()
    state = stepper.step(i, ch1, state)
    _sync(device)
    return state, time.perf_counter() - t0


def time_steps(stepper, ch1, theta0_1, n_steps: int, device: str, cdtype, warmup: int = 3) -> np.ndarray:
    """Seconds per step for one sample (batch 1); warm-up runs the whole sequence, discarded."""
    ch1, state0 = _prepare(stepper, ch1, theta0_1, device, cdtype)
    times = np.empty(n_steps)
    with torch.no_grad():
        for _ in range(warmup):
            s = state0
            for i in range(n_steps):
                s = stepper.step(i, ch1, s)
            _sync(device)
        s = state0
        for i in range(n_steps):
            s, times[i] = _timed_step(stepper, i, ch1, s, device)
    return times


def time_until_converged(ao_stepper, ch1, theta0_1, tol: float, max_iter: int, device: str, cdtype):
    """Run until |dR|/|R| < tol or max_iter. Returns (step_times (n,), wsr_per_step (n+1,)), WSR in bits."""
    ch1, state = _prepare(ao_stepper, ch1, theta0_1, device, cdtype)
    times, wsr = [], []
    with torch.no_grad():
        # warm-up on a throwaway pass so first-call overheads are not timed
        ao_stepper.step(0, ch1, state)
        _sync(device)
        wsr.append(float(fp.wsr(ch1, *state)[0]))
        for i in range(max_iter):
            state, dt = _timed_step(ao_stepper, i, ch1, state, device)
            times.append(dt)
            wsr.append(float(fp.wsr(ch1, *state)[0]))
            if abs(wsr[-1] - wsr[-2]) / max(abs(wsr[-2]), 1e-300) < tol:
                break
    return np.asarray(times), np.asarray(wsr)


def per_iteration_time(stepper, data: dict, idx, device: str, cdtype, n_steps: int = 10) -> np.ndarray:
    """(len(idx),) mean of the first ``n_steps`` step times per sample (batch-1 each)."""
    idx = [int(i) for i in idx]
    out = np.empty(len(idx))
    for j, i in enumerate(idx):
        t = time_steps(stepper, data["ch"].slice([i]), data["theta0"][[i]], n_steps, device, cdtype)
        out[j] = t.mean()
    return out
