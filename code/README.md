# code

```
src/        library modules (flat imports: `import fp_core`, `from models import build_model`)
            config, channels, data, fp_core, ao, teacher, train, timing, plotting, report, runlog,
            atomic, nb_helpers, models/, experiments/
scripts/    make_notebook.py (regenerates master_pipeline.ipynb), run_pipeline.py, build_report.py
tests/      pytest suite (incl. end-to-end notebook test, ~minutes)
artifacts/  datasets, checkpoints, logs, raw results (Config.root); figures + report go to E:/payam/Paper
master_pipeline.ipynb   unexecuted master notebook (first cell adds ./src to sys.path)
```

Run everything from `E:/payam/code`.

- Tests: `E:/payam/env/Scripts/python.exe -m pytest` (pytest.ini sets `pythonpath = src scripts`)
- Notebook: `E:/payam/env/Scripts/python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 master_pipeline.ipynb > run.log 2>&1`
  (or `scripts/run_pipeline.py`, which does the same)
- Regenerate notebook: `E:/payam/env/Scripts/python.exe scripts/make_notebook.py`
- Report only: `E:/payam/env/Scripts/python.exe scripts/build_report.py`
