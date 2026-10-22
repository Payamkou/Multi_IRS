"""Pipeline logging and progress bars.

``get_logger(cfg)`` configures the project logger ``iuf`` once per artifact root:
* a file handler appending to ``{cfg.root}/logs/pipeline.log``
  (format ``%(asctime)s %(levelname)s %(name)s: %(message)s``), so progress survives the session;
* a stdout handler that writes through ``tqdm.write`` (log lines do not break an active bar).
Repeated calls add no duplicate handlers; a call with a different ``cfg.root`` closes the old file
handler and opens the new one (one log file at a time). Module loggers are children (``iuf.train``,
``iuf.sweeps`` ...) and share these handlers; ``iuf`` does not propagate to the root logger.

``bar(iterable, **kw)`` is the classic text tqdm bar (``from tqdm import tqdm``, not the notebook widget)
with pipeline defaults. Bars wrap OUTER loops only (epochs, sweep points, AO chunks, timing samples),
never the inside of a timed region.
"""
from __future__ import annotations

import logging
import os
import sys

from tqdm import tqdm

LOGGER_NAME = "iuf"
LOG_FILE = "pipeline.log"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _TqdmStdoutHandler(logging.Handler):
    """Writes to the *current* sys.stdout (Jupyter swaps it per cell) via tqdm.write."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def log_path(cfg) -> str:
    return os.path.join(cfg.root, "logs", LOG_FILE)


def get_logger(cfg=None, name: str | None = None) -> logging.Logger:
    """The ``iuf`` logger (or its child ``iuf.{name}``), configured for ``cfg.root`` (see module doc).
    With ``cfg=None`` the handlers are left as they are."""
    base = logging.getLogger(LOGGER_NAME)
    base.setLevel(logging.INFO)
    base.propagate = False
    fmt = logging.Formatter(FORMAT)
    if not any(isinstance(h, _TqdmStdoutHandler) for h in base.handlers):
        h = _TqdmStdoutHandler()
        h.setFormatter(fmt)
        base.addHandler(h)
    if cfg is not None:
        path = os.path.abspath(log_path(cfg))
        files = [h for h in base.handlers if isinstance(h, logging.FileHandler)]
        if not any(os.path.normcase(h.baseFilename) == os.path.normcase(path) for h in files):
            for h in files:
                base.removeHandler(h)
                h.close()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = logging.FileHandler(path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            base.addHandler(fh)
    return base.getChild(name) if name else base


def close_logger() -> None:
    """Close and detach the file handler (tests use this before deleting a tmp root)."""
    base = logging.getLogger(LOGGER_NAME)
    for h in [h for h in base.handlers if isinstance(h, logging.FileHandler)]:
        base.removeHandler(h)
        h.close()


def bar(iterable=None, **kw):
    """Classic tqdm text bar with pipeline defaults (stdout, fixed width, >= 1 s between redraws,
    so a notebook executed by nbconvert stores few updates)."""
    kw.setdefault("ncols", 110)
    kw.setdefault("mininterval", 1.0)
    kw.setdefault("file", sys.stdout)   # same stream as the log lines: one coherent output per cell
    kw.setdefault("leave", True)
    return tqdm(iterable, **kw)
