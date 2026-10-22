"""Execute master_pipeline.ipynb in place with nbconvert; stdout/stderr go to run.log.

Run (from anywhere): E:/payam/env/Scripts/python.exe scripts/run_pipeline.py
"""
import os
import subprocess
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    with open(os.path.join(CODE, "run.log"), "w", encoding="utf-8") as log:
        rc = subprocess.call([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                              "--ExecutePreprocessor.timeout=-1", "master_pipeline.ipynb"],
                             cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
    sys.exit(rc)
