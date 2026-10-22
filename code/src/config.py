"""Central configuration, config hashing and the seed registry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class Config:
    N: int = 2
    M: int = 80
    K: int = 6
    L: int = 10
    T: int = 3
    noise_dbm: float = -90.0
    crn_dbm: float = 20.0
    pmn_offset_db: float = -10.0
    crn_weights: tuple = (0.90, 0.80, 0.85)
    pmn_weights: tuple = (0.30, 0.20, 0.25)
    kappa: float = 10.0
    nlos_db: float = 20.0
    shadow_db: float = 8.0
    bs: tuple = (0., 0.)
    disk_center: tuple = (100., 0.)
    disk_radius: float = 20.
    irs_pos: tuple = ((10., 10.), (10., -10.))
    n_train: int = 10000
    n_val: int = 1000
    n_test: int = 1000
    ao_tol: float = 1e-5
    ao_max_iter: int = 100
    n_sup_epochs: int = 50
    n_unsup_epochs: int = 150
    batch_size: int = 200
    lr_grid: tuple = (0.1, 0.01, 0.001)
    pilot_sup_epochs: int = 5
    pilot_unsup_epochs: int = 15
    loss_a: float = 0.5
    loss_version: int = 2   # 2 = Stage-II f_q term (Task 16); bumping it invalidates pilots/checkpoints/results
    grad_clip_norm: float = 1.0
    maso_M_train: int = 80
    tias_hidden: int = 32
    sweep_M: tuple = (20, 40, 60, 80, 100, 120, 140, 160, 180)
    sweep_K: tuple = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
    sweep_crn_dbm: tuple = (20, 25, 30, 35, 40, 45)
    zeroshot_M: tuple = (20, 40, 60, 80, 100, 120, 160)
    timing_threads: int = 6
    timing_warmup: int = 3
    timing_subset: Optional[int] = None
    log_every: int = 15   # console progress summary every N steps (log file gets all); not in any hash key
    train_device: str = "cuda"
    eval_device: str = "cpu"
    root: str = "E:/payam/code/artifacts"
    paper_dir: str = "E:/payam/Paper"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: _tuplify(v) for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_yaml(self, path) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_listify(self.to_dict()), f, sort_keys=False)


def _tuplify(v):
    """Recursively convert lists (from JSON) back into tuples."""
    return tuple(_tuplify(x) for x in v) if isinstance(v, list) else v


def _listify(v):
    if isinstance(v, dict):
        return {k: _listify(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_listify(x) for x in v]
    return v


def config_hash(cfg: Config, keys: tuple) -> str:
    """sha1 of the JSON of the selected fields (order given by ``keys``)."""
    d = cfg.to_dict()
    sel = {k: d[k] for k in keys}
    return hashlib.sha1(json.dumps(sel, sort_keys=True).encode()).hexdigest()


def seed_for(split: str, point=None) -> int:
    """Seed registry.

    train 1001, val 2001, test 3001; sweep_M 40000+M; sweep_K 50000+K;
    sweep_power 60000+int(dBm); model_init 9001.

    Not handled here: the initial-phase seed ("init_phases"). Each dataset
    derives its phase seed as its own seed + 7.
    """
    fixed = {"train": 1001, "val": 2001, "test": 3001, "model_init": 9001}
    if split in fixed:
        return fixed[split]
    base = {"sweep_M": 40000, "sweep_K": 50000, "sweep_power": 60000}
    if split in base:
        if point is None:
            raise ValueError(f"{split} requires a point")
        return base[split] + int(point)
    raise KeyError(f"unknown split {split!r}")
