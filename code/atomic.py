"""Atomic-write helper: ``os.replace`` with a short retry loop.

On Windows a just-written file can be held open briefly by an antivirus scanner, the indexer or a
reader in another process, making ``os.replace`` fail with PermissionError (WinError 5/32). Every
atomic writer in the project (train, teacher, data, experiments, report) goes through ``replace``.
"""
from __future__ import annotations

import os
import time

TRIES = 5


def replace(src, dst, tries: int = TRIES, delay: float = 0.2) -> None:
    """``os.replace(src, dst)``, tried up to ``tries`` times with a short sleep (``delay`` s, doubling)
    between attempts on OSError; the last error is re-raised."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(delay * (2 ** i))
