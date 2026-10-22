"""Spec §8: training losses (supervised phase alignment, hybrid unsupervised, deep supervision)."""
import dataclasses

import pytest
import torch

import ao
import data
import fp_core as fp
from channels import generate_channels
from config import Config
from models.fonr import FONR
from train import loss_deep_supervision, loss_supervised, loss_unsup_layer, rho_weights

C128, F64 = torch.complex128, torch.float64


def _make(B=4, M=16, K=6, seed=3, cdtype=C128):
    cfg = dataclasses.replace(Config(), M=M, K=K)
    ch, theta0 = generate_channels(cfg, B, seed=seed, M=M, K=K)
    ch = ch.to("cpu", cdtype)
    p0, phi0 = ao.init_state(ch, theta0)
    return ch, p0, phi0


def test_supervised_loss_zero_at_label():
    _, _, phi = _make()
    assert float(loss_supervised(phi, phi)) == pytest.approx(0.0, abs=1e-12)


def test_supervised_loss_two_at_antipode():
    _, _, phi = _make()
    assert float(loss_supervised(-phi, phi)) == pytest.approx(2.0, abs=1e-12)


def test_supervised_loss_is_mean_over_bnm():
    _, _, phi = _make()
    gen = torch.Generator().manual_seed(1)
    d = torch.rand(phi.shape, generator=gen, dtype=F64)
    got = loss_supervised(phi * torch.exp(1j * d), phi)
    assert float(got) == pytest.approx(float((1 - torch.cos(d)).mean()), rel=1e-12)


def test_unsup_no_grad_through_alpha():
    ch, p0, phi0 = _make()
    p = p0.clone().requires_grad_(True)
    phi = phi0.clone().requires_grad_(True)
    alpha_in = fp.aux_vars(ch, p0, phi0).alpha.clone().requires_grad_(True)
    loss_unsup_layer(ch, p, phi, alpha_in).backward()
    assert alpha_in.grad is None
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert phi.grad is not None and torch.isfinite(phi.grad).all() and phi.grad.abs().sum() > 0


def test_unsup_nats():
    ch, p, phi = _make()
    alpha = fp.aux_vars(ch, p, phi).alpha
    got = loss_unsup_layer(ch, p, phi, alpha, a=1.0)
    want = -fp.wsr(ch, p, phi, base="e").mean()
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_unsup_f3_equals_ln_wsr_at_alpha_gamma():
    """Composite = mean(-a R_nats - (1-a) f3), eps closed-form at the given state."""
    ch, p, phi = _make()
    aux = fp.aux_vars(ch, p, phi)
    f3 = fp.surrogate_f3(ch, p, phi, aux.wt, aux.eps)
    got = loss_unsup_layer(ch, p, phi, aux.alpha, a=0.25)
    want = (-0.25 * fp.wsr(ch, p, phi, base="e") - 0.75 * f3).mean()
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_rho_weights_sum_to_one():
    rho = rho_weights(10)
    assert rho.shape == (10,)
    assert float(rho.sum()) == pytest.approx(1.0, abs=1e-15)
    assert torch.allclose(rho, torch.arange(1, 11, dtype=F64) / 55.0)
    assert torch.all(rho[1:] > rho[:-1])


def test_deep_supervision_is_rho_weighted_sum():
    ch, p0, phi0 = _make()
    net = FONR(L=10, T=3, mu_init=1e-3)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
        got = loss_deep_supervision(ch, tr, a=0.5)
        want = sum((l / 55.0) * loss_unsup_layer(ch, tr.p[l], tr.phi[l], tr.alpha_in[l - 1], 0.5)
                   for l in range(1, 11))
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_deep_supervision_grads_finite_complex64_zero_power_user():
    ch, p0, phi0 = _make(cdtype=torch.complex64)
    p0 = p0.clone()
    p0[:, 2] = 0.0
    net = FONR(L=10, T=3, mu_init=1e-3)
    tr = net(ch, p0, phi0)
    loss = loss_deep_supervision(ch, tr)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(net.mu.grad).all() and net.mu.grad.abs().sum() > 0


# ================================================================ training loop (Task 10)
import csv
import json
import math
import os
import shutil
import time

import teacher
import train

HIST_COLS = ["epoch", "stage", "lr", "train_loss", "val_loss", "train_wsr_mean", "train_wsr_std",
             "val_wsr_mean", "val_wsr_std", "val_phase_cosine_sim", "is_best", "epoch_time_sec",
             "n_skipped_batches"]


def _tiny_cfg(root, **kw):
    base = dict(n_train=64, n_val=16, n_test=16, M=16, n_sup_epochs=2, n_unsup_epochs=2,
                batch_size=32, train_device="cpu", eval_device="cpu", lr_grid=(0.01,),
                pilot_sup_epochs=1, pilot_unsup_epochs=1, ao_max_iter=20, root=str(root))
    base.update(kw)
    return dataclasses.replace(Config(), **base)


@pytest.fixture(scope="module")
def base_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("train_base")
    teacher.build_teacher_labels(_tiny_cfg(root), chunk=64)
    return root


@pytest.fixture
def fresh_root(base_root, tmp_path):
    """A new artifacts root that shares the (copied) tiny datasets and teacher labels."""
    def make(tag):
        r = tmp_path / tag
        shutil.copytree(base_root / "datasets", r / "datasets")
        if (base_root / "logs").exists():
            shutil.copytree(base_root / "logs", r / "logs")
        return r
    return make


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("name", ["fonr", "tias"])
def test_resume_is_exact(name, fresh_root):
    cfg_a = _tiny_cfg(fresh_root("a"))
    cfg_b = _tiny_cfg(fresh_root("b"))
    pa = train.train_model(name, cfg_a)
    train.train_model(name, cfg_b, stop_after_epoch=2)
    ck_mid = torch.load(os.path.join(cfg_b.root, "checkpoints", f"{name}_last.pt"), weights_only=False)
    assert ck_mid["epoch"] == 2
    pb = train.train_model(name, cfg_b)
    la = torch.load(os.path.join(cfg_a.root, "checkpoints", f"{name}_last.pt"), weights_only=False)
    lb = torch.load(os.path.join(cfg_b.root, "checkpoints", f"{name}_last.pt"), weights_only=False)
    assert la["epoch"] == lb["epoch"] == 4
    for k, v in la["model_state"].items():  # bit-exact and finite
        assert torch.isfinite(v).all(), k
        assert torch.equal(v, lb["model_state"][k]), k
    for k, v in la["optimizer_state"]["state"].items():
        for kk, vv in v.items():
            assert torch.equal(vv, lb["optimizer_state"]["state"][k][kk]), (k, kk)
    ba = torch.load(pa, weights_only=False)["model_state"]
    bb = torch.load(pb, weights_only=False)["model_state"]
    for k, v in ba.items():
        assert torch.equal(v, bb[k]), k
    ha = _read_csv(os.path.join(cfg_a.root, "logs", f"{name}_training_history.csv"))
    hb = _read_csv(os.path.join(cfg_b.root, "logs", f"{name}_training_history.csv"))
    assert len(ha) == len(hb) == 4
    for ra, rb in zip(ha, hb):
        ra.pop("epoch_time_sec"), rb.pop("epoch_time_sec")
        assert ra == rb


def test_completed_training_not_rerun(fresh_root):
    cfg = _tiny_cfg(fresh_root("c"))
    best = train.train_model("fonr", cfg)
    last = os.path.join(cfg.root, "checkpoints", "fonr_last.pt")
    m1 = os.path.getmtime(last)
    time.sleep(0.05)
    assert train.train_model("fonr", cfg) == best
    assert os.path.getmtime(last) == m1


def test_resume_refuses_mismatched_config(fresh_root):
    cfg = _tiny_cfg(fresh_root("d"))
    train.train_model("fonr", cfg, stop_after_epoch=1)
    with pytest.raises(ValueError, match="config mismatch"):
        train.train_model("fonr", dataclasses.replace(cfg, batch_size=16))


def test_history_columns_and_checkpoint_keys(fresh_root):
    cfg = _tiny_cfg(fresh_root("e"))
    best = train.train_model("tias", cfg)
    with open(os.path.join(cfg.root, "logs", "tias_training_history.csv"), newline="") as f:
        header = next(csv.reader(f))
    assert header == HIST_COLS
    rows = _read_csv(os.path.join(cfg.root, "logs", "tias_training_history.csv"))
    assert [r["stage"] for r in rows] == ["supervised", "supervised", "unsupervised", "unsupervised"]
    ck = torch.load(best, weights_only=False)
    for k in ("model_name", "epoch", "stage", "model_state", "optimizer_state", "selected_lr",
              "best_val_wsr", "best_epoch", "rng_cpu", "rng_cuda", "metrics_history", "config", "cfg_hash"):
        assert k in ck, k
    assert ck["cfg_hash"] == train.config_hash(cfg, train.TRAIN_KEYS)
    assert ck["selected_lr"] == 0.01


def test_no_test_metrics_in_history(fresh_root):
    cfg = _tiny_cfg(fresh_root("f"))
    best = train.train_model("tias", cfg)
    last = os.path.join(cfg.root, "checkpoints", "tias_last.pt")
    for path in (best, last):
        mh = torch.load(path, weights_only=False)["metrics_history"]
        assert mh and not any(k.startswith("test") for k in mh)
        assert any(k.startswith("train") for k in mh) and any(k.startswith("val") for k in mh)


def test_lr_pilot_csv_and_reuse(fresh_root):
    cfg = _tiny_cfg(fresh_root("g"), lr_grid=(0.01, 0.001))
    lr = train.run_lr_pilot("tias", cfg)
    path = os.path.join(cfg.root, "logs", "tias_lr_pilot.csv")
    rows = _read_csv(path)
    assert list(rows[0].keys()) == ["lr", "epoch", "stage", "val_wsr_mean"]
    assert len(rows) == 2 * (cfg.pilot_sup_epochs + cfg.pilot_unsup_epochs)
    final = {float(r["lr"]): float(r["val_wsr_mean"]) for r in rows if int(r["epoch"]) == 2}
    assert lr == max(final, key=final.get)
    m1 = os.path.getmtime(path)
    time.sleep(0.05)
    assert train.run_lr_pilot("tias", cfg) == lr
    with pytest.raises(ValueError, match="config mismatch"):
        train.run_lr_pilot("tias", dataclasses.replace(cfg, pilot_unsup_epochs=2))
    assert os.path.getmtime(path) == m1


def test_evaluate_complex128_default(fresh_root):
    cfg = _tiny_cfg(fresh_root("h"))
    d = data.build_or_load(cfg, "val")
    t = teacher.load_teacher(cfg, "val")
    from models import build_model
    net = build_model("tias", cfg)
    r = train.evaluate(net, d["ch"], d["theta0"], phi_star=t["phi_star"], a=0.5)
    assert r["wsr_per_sample"].dtype == torch.float64 and r["wsr_per_sample"].shape == (16,)
    assert next(net.parameters()).dtype == torch.float32  # caller's model untouched
    assert math.isfinite(r["loss"]) and -1.0 <= r["phase_cos"] <= 1.0
    r32 = train.evaluate(net, d["ch"], d["theta0"], phi_star=t["phi_star"], cdtype=torch.complex64)
    assert r32["wsr_per_sample"].dtype == torch.float32
    assert r32["wsr_mean"] == pytest.approx(r["wsr_mean"], rel=1e-3)


def test_final_metrics_row(fresh_root):
    cfg = _tiny_cfg(fresh_root("i"))
    train.train_model("tias", cfg)
    out = train.final_metrics("tias", cfg)
    rows = _read_csv(os.path.join(cfg.root, "logs", "all_models_final_metrics.csv"))
    assert len(rows) == 1
    assert list(rows[0].keys()) == train.FINAL_COLS
    cols = ["model_name", "selected_lr", "best_epoch"]
    for s in ("train", "val", "test"):
        cols += [f"{s}_{c}" for c in ("loss", "wsr_mean", "wsr_std", "ao_wsr_mean", "efficiency_pct")]
    cols += ["test_gap_median", "test_gap_95th_pctl", "test_gap_min_left_tail"]
    assert set(cols) <= set(rows[0].keys())
    assert math.isfinite(float(rows[0]["test_gap_95th_pctl"]))
    assert out["test_efficiency_pct"] == pytest.approx(100 * out["test_wsr_mean"] / out["test_ao_wsr_mean"])
    with open(os.path.join(cfg.root, "logs", "tias_final_metrics.json")) as f:
        assert json.load(f)["model_name"] == "tias"
    assert all(math.isfinite(float(rows[0][c])) for c in train.FINAL_COLS[1:])
    train.final_metrics("tias", cfg)  # upsert, not append
    assert len(_read_csv(os.path.join(cfg.root, "logs", "all_models_final_metrics.csv"))) == 1


def test_grad_clip_norm_in_train_keys():
    assert Config().grad_clip_norm == 1.0
    assert "grad_clip_norm" in train.TRAIN_KEYS


def test_nonfinite_gradient_step_is_skipped():
    torch.manual_seed(0)
    net = torch.nn.Linear(3, 1)
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    x = torch.randn(4, 3)
    assert train._optimizer_step(net, opt, net(x).pow(2).mean(), 1.0) is False  # normal step
    before = {k: v.clone() for k, v in net.state_dict().items()}
    st_before = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in opt.state_dict()["state"].items()}
    bad = net(x).sum() * torch.tensor(float("nan"))
    assert train._optimizer_step(net, opt, bad, 1.0) is True
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k])
    for k, v in opt.state_dict()["state"].items():
        for kk, vv in v.items():
            assert torch.equal(vv, st_before[k][kk])
    assert all(p.grad is None for p in net.parameters())
    inf_grad = net(x * float("inf")).sum()  # finite-or-not loss with inf grads
    assert train._optimizer_step(net, opt, inf_grad, 1.0) is True


def test_grad_clipped_to_max_norm():
    net = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    w0 = [p.detach().clone() for p in net.parameters()]
    loss = 1e6 * net(torch.ones(1, 3)).sum()
    train._optimizer_step(net, opt, loss, 1.0)
    step = torch.cat([(p.detach() - w).flatten() for p, w in zip(net.parameters(), w0)])
    assert float(step.norm()) == pytest.approx(1.0, rel=1e-5)


def test_epoch_counts_skipped_batches(fresh_root, monkeypatch):
    cfg = _tiny_cfg(fresh_root("j"))
    D = train._Data(cfg, "cpu")
    net = train._fresh_model("tias", cfg, "cpu")
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    w0 = {k: v.clone() for k, v in net.state_dict().items()}
    monkeypatch.setattr(train, "loss_supervised", lambda a, b: (a * b.conj()).real.mean() * float("nan"))
    m = train._train_epoch(net, opt, D, cfg, 1, cfg.n_sup_epochs)
    assert m["n_skipped_batches"] == 2
    for k, v in net.state_dict().items():
        assert torch.equal(v, w0[k])
    assert opt.state_dict()["state"] == {}


def test_fonr_tiny_training_finite_with_best(fresh_root):
    cfg = _tiny_cfg(fresh_root("k"))
    best = train.train_model("fonr", cfg)
    assert os.path.exists(best)
    ck = torch.load(best, weights_only=False)
    assert all(torch.isfinite(v).all() for v in ck["model_state"].values())
    assert math.isfinite(ck["best_val_wsr"])
    rows = _read_csv(os.path.join(cfg.root, "logs", "fonr_training_history.csv"))
    assert all(math.isfinite(float(r["val_wsr_mean"])) for r in rows)
    last = torch.load(os.path.join(cfg.root, "checkpoints", "fonr_last.pt"), weights_only=False)
    assert all(torch.isfinite(v).all() for v in last["model_state"].values())
    assert last["metrics_history"]["n_skipped_batches"] == [int(r["n_skipped_batches"]) for r in rows]


def test_epoch_means_exclude_skipped_batches(fresh_root, monkeypatch):
    cfg = _tiny_cfg(fresh_root("l"))
    D = train._Data(cfg, "cpu")
    net = train._fresh_model("tias", cfg, "cpu")
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    real = train.loss_supervised
    calls = []

    def first_nan(a, b):
        calls.append(1)
        out = real(a, b)
        return out * float("nan") if len(calls) == 1 else out

    monkeypatch.setattr(train, "loss_supervised", first_nan)
    m = train._train_epoch(net, opt, D, cfg, 1, cfg.n_sup_epochs)
    assert m["n_skipped_batches"] == 1
    assert math.isfinite(m["train_loss"]) and math.isfinite(m["train_wsr_mean"])
    assert math.isfinite(m["train_wsr_std"])
    monkeypatch.setattr(train, "loss_supervised", lambda a, b: real(a, b) * float("nan"))
    m = train._train_epoch(net, opt, D, cfg, 1, cfg.n_sup_epochs)
    assert m["n_skipped_batches"] == 2 and math.isnan(m["train_loss"]) and math.isnan(m["train_wsr_mean"])


def test_pilot_csv_without_sidecar_refused(fresh_root):
    cfg = _tiny_cfg(fresh_root("m"))
    train.run_lr_pilot("tias", cfg)
    os.remove(os.path.join(cfg.root, "logs", "tias_lr_pilot.json"))
    with pytest.raises(ValueError, match="sidecar"):
        train.run_lr_pilot("tias", cfg)


def test_finished_run_without_best_raises(fresh_root):
    cfg = _tiny_cfg(fresh_root("n"))
    best = train.train_model("tias", cfg)
    os.remove(best)
    with pytest.raises(RuntimeError, match="best.pt"):
        train.train_model("tias", cfg)
