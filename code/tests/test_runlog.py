"""Task 16 B: pipeline logger (file + stdout, idempotent) and classic tqdm bars."""
import dataclasses
import logging
import os

import runlog
from config import Config


def _cfg(root):
    return dataclasses.replace(Config(), root=str(root))


def test_logger_writes_to_pipeline_log(tmp_path):
    cfg = _cfg(tmp_path)
    try:
        log = runlog.get_logger(cfg, "train")
        log.info("hello %d", 42)
        path = runlog.log_path(cfg)
        assert path == os.path.join(str(tmp_path), "logs", "pipeline.log")
        for h in logging.getLogger("iuf").handlers:
            h.flush()
        text = open(path, encoding="utf-8").read()
        assert "INFO iuf.train: hello 42" in text
        log.info("second")                               # append, not overwrite
        for h in logging.getLogger("iuf").handlers:
            h.flush()
        assert open(path, encoding="utf-8").read().startswith(text)
    finally:
        runlog.close_logger()


def test_no_duplicate_handlers(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    try:
        for _ in range(4):
            runlog.get_logger(cfg)
            runlog.get_logger(cfg, "sweeps")
        base = logging.getLogger("iuf")
        assert sum(isinstance(h, logging.FileHandler) for h in base.handlers) == 1
        assert sum(isinstance(h, runlog._TqdmStdoutHandler) for h in base.handlers) == 1
        assert base.propagate is False
        runlog.get_logger(cfg, "x").info("once")
        out = capsys.readouterr().out
        assert out.count("once") == 1
        cfg2 = _cfg(tmp_path / "other")                  # new root: the file handler is swapped
        runlog.get_logger(cfg2).info("moved")
        files = [h for h in base.handlers if isinstance(h, logging.FileHandler)]
        assert len(files) == 1 and files[0].baseFilename == os.path.abspath(runlog.log_path(cfg2))
    finally:
        runlog.close_logger()


def test_bar_is_classic_tqdm():
    from tqdm import tqdm
    b = runlog.bar(range(3), desc="x", disable=True)
    assert type(b) is tqdm
    assert list(b) == [0, 1, 2]
