"""Rebuild results_findings.md from the cached experiment results (default Config paths).

Run: E:/payam/env/Scripts/python.exe scripts/build_report.py [ao_max_iter]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import nb_helpers as H  # noqa: E402
import report  # noqa: E402

if __name__ == "__main__":
    cfg = H.make_config(int(sys.argv[1]) if len(sys.argv) > 1 else H.TABLE_I_AO_MAX_ITER)
    print(report.build_report(cfg))
