"""Training (spec §8).

Training section: ``run_lr_pilot``, ``train_model`` (resumable, CSV history, last/best
checkpoints), ``evaluate`` (no-grad, default complex128/CPU), ``final_metrics``.

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

import copy
import csv
import json
import math
import os
import time
import warnings
from pathlib import Path

import torch

import ao
import data
import fp_core as fp
import teacher
from channels import ChannelBatch
from config import Config, config_hash, seed_for
from models import build_model
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


# ================================================================ training loop (spec §8)
TRAIN_KEYS = ("N", "M", "K", "L", "T", "n_train", "n_sup_epochs", "n_unsup_epochs", "batch_size",
              "lr_grid", "loss_a", "maso_M_train", "tias_hidden", "grad_clip_norm")
PILOT_KEYS = TRAIN_KEYS + ("pilot_sup_epochs", "pilot_unsup_epochs")

HISTORY_COLS = ["epoch", "stage", "lr", "train_loss", "val_loss", "train_wsr_mean", "train_wsr_std",
                "val_wsr_mean", "val_wsr_std", "val_phase_cosine_sim", "is_best", "epoch_time_sec",
                "n_skipped_batches"]  # spec §8 columns + one trailing column (controller ruling)
PILOT_COLS = ["lr", "epoch", "stage", "val_wsr_mean"]
FINAL_COLS = (["model_name", "selected_lr", "best_epoch"]
              + [f"{s}_{c}" for s in ("train", "val", "test")
                 for c in ("loss", "wsr_mean", "wsr_std", "ao_wsr_mean", "efficiency_pct")]
              + ["test_gap_median", "test_gap_95th_pctl", "test_gap_min_left_tail"])

C64, C128 = torch.complex64, torch.complex128


# ---------------------------------------------------------------- helpers
def _dirs(cfg: Config):
    ck = os.path.join(cfg.root, "checkpoints")
    logs = os.path.join(cfg.root, "logs")
    os.makedirs(ck, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    return ck, logs


def _train_device(cfg: Config) -> str:
    dev = cfg.train_device
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA unavailable: training falls back to CPU")
        print("WARNING: CUDA unavailable, training on CPU", flush=True)
        return "cpu"
    return dev


def _rdtype(cdtype):
    return torch.float32 if cdtype == C64 else torch.float64


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_csv(path, cols, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    os.replace(tmp, path)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stage(epoch: int, n_sup: int) -> str:
    return "supervised" if epoch <= n_sup else "unsupervised"


# ---------------------------------------------------------------- evaluation
def evaluate(model, ch: ChannelBatch, theta0: torch.Tensor, phi_star=None, a: float = 0.5,
             device=None, cdtype=None, chunk: int = 1000) -> dict:
    """No-grad evaluation of the L-layer output on a whole split.

    Defaults: complex128 on CPU (final metrics). Selection-time evaluation passes the
    training device and complex64. The caller's model is never modified (a cast copy is used,
    so float32 parameters become float64 and the whole forward runs in complex128).
    loss = loss_deep_supervision (the Stage-II loss); phase_cos = mean Re{phi^(L) conj(phi*)}.
    """
    device = device or "cpu"
    cdtype = cdtype or C128
    net = copy.deepcopy(model).to(device=device, dtype=_rdtype(cdtype)).eval()
    B = ch.B
    loss_sum, cos_sum, wsrs = 0.0, 0.0, []
    with torch.no_grad():
        for lo in range(0, B, chunk):
            sl = slice(lo, min(lo + chunk, B))
            c = ch.slice(sl).to(device, cdtype)
            p0, phi0 = ao.init_state(c, theta0[sl])
            tr = net(c, p0, phi0)
            n = c.B
            loss_sum += float(loss_deep_supervision(c, tr, a)) * n
            wsrs.append(fp.wsr(c, tr.p[-1], tr.phi[-1]).cpu())
            if phi_star is not None:
                ps = phi_star[sl].to(device=device, dtype=cdtype)
                cos_sum += float((tr.phi[-1] * ps.conj()).real.mean()) * n
    w = torch.cat(wsrs)
    return {"loss": loss_sum / B, "wsr_mean": float(w.mean()), "wsr_std": float(w.std()),
            "wsr_per_sample": w, "phase_cos": (cos_sum / B) if phi_star is not None else float("nan")}


# ---------------------------------------------------------------- shared epoch machinery
class _Data:
    """Train split on the training device at complex64 (+ val split kept on CPU for evaluate)."""

    def __init__(self, cfg: Config, device):
        tr = data.build_or_load(cfg, "train")
        va = data.build_or_load(cfg, "val")
        self.ch = tr["ch"].to(device, C64)
        self.p0, self.phi0 = ao.init_state(self.ch, tr["theta0"])
        self.phi_star = teacher.load_teacher(cfg, "train")["phi_star"].to(device=device, dtype=C64)
        self.ch_val, self.theta_val = va["ch"], va["theta0"]
        self.phi_star_val = teacher.load_teacher(cfg, "val")["phi_star"]
        self.device = device
        self.B = self.ch.B


def _fresh_model(name: str, cfg: Config, device):
    torch.manual_seed(seed_for("model_init"))
    return build_model(name, cfg).to(device)


def _optimizer_step(model, opt, loss: torch.Tensor, max_norm: float) -> bool:
    """Backward + guarded step, identical for every model. Returns True if the batch was skipped.

    Non-finite loss or any non-finite parameter gradient -> grads cleared, no optimizer step
    (Adam state untouched). Otherwise clip the global grad norm to ``max_norm`` and step."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in model.parameters() if p.grad is not None]
    ok = bool(torch.isfinite(loss.detach())) and all(bool(torch.isfinite(p.grad).all()) for p in params)
    if not ok:
        opt.zero_grad(set_to_none=True)
        return True
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    opt.step()
    return False


def _train_epoch(model, opt, D: _Data, cfg: Config, epoch: int, n_sup: int) -> dict:
    """One epoch; epochs 1..n_sup supervised, later deep-supervised Stage II.
    train_* are means over the epoch's NON-skipped batches (train_wsr_std = mean of per-batch
    stds); NaN if every batch was skipped. n_skipped_batches counts batches whose step was
    skipped for non-finite loss/gradients."""
    model.train()
    gen = torch.Generator().manual_seed(seed_for("train") + epoch)
    perm = torch.randperm(D.B, generator=gen)
    losses, means, stds, n_skip = [], [], [], 0
    for lo in range(0, D.B, cfg.batch_size):
        idx = perm[lo:lo + cfg.batch_size].to(D.device)
        c = D.ch.slice(idx)
        tr = model(c, D.p0[idx], D.phi0[idx])
        if epoch <= n_sup:
            loss = loss_supervised(tr.phi[-1], D.phi_star[idx])
        else:
            loss = loss_deep_supervision(c, tr, cfg.loss_a)
        skipped = _optimizer_step(model, opt, loss, cfg.grad_clip_norm)
        n_skip += skipped
        if skipped:
            continue
        with torch.no_grad():
            w = fp.wsr(c, tr.p[-1].detach(), tr.phi[-1].detach())
        losses.append(loss.item())
        means.append(float(w.mean()))
        stds.append(float(w.std()) if w.numel() > 1 else 0.0)
    nb = len(losses)
    avg = (lambda v: sum(v) / nb) if nb else (lambda v: float("nan"))
    return {"train_loss": avg(losses), "train_wsr_mean": avg(means),
            "train_wsr_std": avg(stds), "n_skipped_batches": int(n_skip)}


def _val(model, D: _Data, cfg: Config) -> dict:
    """Selection-time validation at the training device and precision."""
    return evaluate(model, D.ch_val, D.theta_val, D.phi_star_val, a=cfg.loss_a, device=D.device, cdtype=C64)


# ---------------------------------------------------------------- LR pilot
def _pilot_choice(rows) -> float:
    last = {}
    for r in rows:
        lr, ep = float(r["lr"]), int(r["epoch"])
        if lr not in last or ep > last[lr][0]:
            v = float(r["val_wsr_mean"])
            last[lr] = (ep, v if math.isfinite(v) else -math.inf)
    best = None
    for lr, (_, v) in last.items():  # rows are in grid order; ties keep the first
        if best is None or v > last[best][1]:
            best = lr
    return best


def run_lr_pilot(name: str, cfg: Config) -> float:
    """Short pilot per lr in cfg.lr_grid (fresh model, seed model_init); pick the best
    final mean validation WSR. Reuses {name}_lr_pilot.csv if present."""
    name = name.lower()
    _, logs = _dirs(cfg)
    path = os.path.join(logs, f"{name}_lr_pilot.csv")
    meta_path = os.path.join(logs, f"{name}_lr_pilot.json")
    h = config_hash(cfg, PILOT_KEYS)
    if os.path.exists(path):
        if not os.path.exists(meta_path):
            raise ValueError(f"{path} exists without its sidecar {meta_path}; cannot verify the config "
                             "(delete the CSV to rerun the pilot)")
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f).get("cfg_hash") != h:
                raise ValueError("config mismatch")
        return _pilot_choice(_read_csv(path))
    device = _train_device(cfg)
    D = _Data(cfg, device)
    n_sup, n_tot = cfg.pilot_sup_epochs, cfg.pilot_sup_epochs + cfg.pilot_unsup_epochs
    rows = []
    for lr in cfg.lr_grid:
        model = _fresh_model(name, cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, n_tot + 1):
            t0 = time.time()
            m = _train_epoch(model, opt, D, cfg, ep, n_sup)
            v = _val(model, D, cfg)
            st = _stage(ep, n_sup)
            rows.append({"lr": lr, "epoch": ep, "stage": st, "val_wsr_mean": v["wsr_mean"]})
            print(f"[pilot {name} lr={lr:g}] epoch {ep}/{n_tot} {st} train_loss={m['train_loss']:.5g} "
                  f"val_wsr={v['wsr_mean']:.4f} skipped={m['n_skipped_batches']} "
                  f"t={time.time() - t0:.1f}s", flush=True)
    choice = _pilot_choice(rows)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"cfg_hash": h, "selected_lr": choice, "lr_grid": list(cfg.lr_grid)}, f, indent=2)
    _write_csv(path, PILOT_COLS, rows)  # written last: its existence marks a completed pilot
    return choice


# ---------------------------------------------------------------- full training
def _history_rows(mh: dict):
    return [{c: mh[c][i] for c in HISTORY_COLS} for i in range(len(mh["epoch"]))]


def train_model(name: str, cfg: Config, resume: bool = True, force_retrain: bool = False,
                stop_after_epoch: int | None = None) -> Path:
    """Two-stage training (spec §8) with exact resume. Returns the path of {name}_best.pt."""
    name = name.lower()
    ckdir, logs = _dirs(cfg)
    last_p = os.path.join(ckdir, f"{name}_last.pt")
    best_p = os.path.join(ckdir, f"{name}_best.pt")
    hist_p = os.path.join(logs, f"{name}_training_history.csv")
    h = config_hash(cfg, TRAIN_KEYS)
    n_sup, n_tot = cfg.n_sup_epochs, cfg.n_sup_epochs + cfg.n_unsup_epochs

    ck = None
    if resume and not force_retrain and os.path.exists(last_p):
        ck = torch.load(last_p, weights_only=False, map_location="cpu")
        if ck["cfg_hash"] != h:
            raise ValueError("config mismatch")
        if ck["epoch"] >= n_tot:
            if not os.path.exists(best_p):
                raise RuntimeError(f"{name}: training finished but {best_p} (best.pt) is missing "
                                   "(no finite validation WSR, or file deleted); use force_retrain=True")
            return Path(best_p)

    device = _train_device(cfg)
    D = _Data(cfg, device)
    if ck is None:
        lr = run_lr_pilot(name, cfg)
        model = _fresh_model(name, cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        start, best_wsr, best_epoch = 1, -math.inf, 0
        mh = {c: [] for c in HISTORY_COLS}
    else:
        lr = ck["selected_lr"]
        model = build_model(name, cfg).to(device)
        model.load_state_dict(ck["model_state"])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        opt.load_state_dict(ck["optimizer_state"])
        torch.set_rng_state(ck["rng_cpu"])
        if ck["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        start, best_wsr, best_epoch = ck["epoch"] + 1, ck["best_val_wsr"], ck["best_epoch"]
        mh = ck["metrics_history"]
        print(f"[{name}] resuming at epoch {start}", flush=True)

    for ep in range(start, n_tot + 1):
        t0 = time.time()
        st = _stage(ep, n_sup)
        m = _train_epoch(model, opt, D, cfg, ep, n_sup)
        v = _val(model, D, cfg)
        is_best = math.isfinite(v["wsr_mean"]) and v["wsr_mean"] > best_wsr
        if is_best:
            best_wsr, best_epoch = v["wsr_mean"], ep
        row = {"epoch": ep, "stage": st, "lr": lr, **m, "val_loss": v["loss"],
               "val_wsr_mean": v["wsr_mean"], "val_wsr_std": v["wsr_std"],
               "val_phase_cosine_sim": v["phase_cos"], "is_best": int(is_best),
               "epoch_time_sec": time.time() - t0}
        for c in HISTORY_COLS:
            mh[c].append(row[c])
        state = {"model_name": name, "epoch": ep, "stage": st,
                 "model_state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
                 "optimizer_state": opt.state_dict(), "selected_lr": lr,
                 "best_val_wsr": best_wsr, "best_epoch": best_epoch,
                 "rng_cpu": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "metrics_history": mh, "config": cfg.to_dict(), "cfg_hash": h}
        if is_best:
            _atomic_save(state, best_p)
        _atomic_save(state, last_p)
        _write_csv(hist_p, HISTORY_COLS, _history_rows(mh))
        mark = " *" if is_best else ""
        print(f"[{name}] epoch {ep}/{n_tot} {st} train_loss={m['train_loss']:.5g} "
              f"val_wsr={v['wsr_mean']:.4f}{mark} skipped={m['n_skipped_batches']} "
              f"t={row['epoch_time_sec']:.1f}s", flush=True)
        if stop_after_epoch is not None and ep >= stop_after_epoch:
            break
    if not os.path.exists(best_p):
        msg = f"{name}: no finite validation WSR so far; {best_p} (best.pt) not written"
        if mh["epoch"] and mh["epoch"][-1] >= n_tot:
            raise RuntimeError(msg)
        warnings.warn(msg)
    return Path(best_p)


# ---------------------------------------------------------------- final metrics
def final_metrics(name: str, cfg: Config) -> dict:
    """Evaluate best.pt on train/val/test in complex128 on cfg.eval_device (the only test use)."""
    name = name.lower()
    ckdir, logs = _dirs(cfg)
    ck = torch.load(os.path.join(ckdir, f"{name}_best.pt"), weights_only=False, map_location="cpu")
    model = build_model(name, cfg)
    model.load_state_dict(ck["model_state"])
    out = {"model_name": name, "selected_lr": ck["selected_lr"], "best_epoch": ck["best_epoch"]}
    for s in ("train", "val", "test"):
        d = data.build_or_load(cfg, s)
        t = teacher.load_teacher(cfg, s)
        r = evaluate(model, d["ch"], d["theta0"], t["phi_star"], a=cfg.loss_a,
                     device=cfg.eval_device, cdtype=C128)
        ao_w = t["wsr_ao"].to(torch.float64)
        out[f"{s}_loss"] = r["loss"]
        out[f"{s}_wsr_mean"] = r["wsr_mean"]
        out[f"{s}_wsr_std"] = r["wsr_std"]
        out[f"{s}_ao_wsr_mean"] = float(ao_w.mean())
        out[f"{s}_efficiency_pct"] = 100.0 * r["wsr_mean"] / float(ao_w.mean())
        if s == "test":
            gap = ao_w - r["wsr_per_sample"].to(torch.float64)
            out["test_gap_median"] = float(torch.quantile(gap, 0.5))
            out["test_gap_95th_pctl"] = float(torch.quantile(gap, 0.95))
            out["test_gap_min_left_tail"] = float(gap.min())
    with open(os.path.join(logs, f"{name}_final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    csv_p = os.path.join(logs, "all_models_final_metrics.csv")
    rows = [r for r in (_read_csv(csv_p) if os.path.exists(csv_p) else []) if r["model_name"] != name]
    rows.append(out)
    _write_csv(csv_p, FINAL_COLS, rows)
    return out
