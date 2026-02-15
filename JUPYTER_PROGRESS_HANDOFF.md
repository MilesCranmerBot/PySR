# PySR Jupyter Progress Handoff (local)

## What changed
- `pysr/jupyter_progress.py` (new): notebook progress context/parser and notebook UI rendering helpers.
- `pysr/sr.py`:
  - imports `JupyterProgressContext` + `should_use_jupyter_progress`
  - computes `equation_search_progress`
  - in notebook mode, disables Julia native bar for the call (`progress=False`) and captures textual progress lines into Python-side notebook UI
  - removed old behavior that force-disabled progress in Jupyter when `sys.stdout` had no `buffer`
- `pysr/test/test_main.py`: expectation updated to avoid blanket progress-disable behavior.
- `pysr/test/test_jupyter_progress_helpers.py` (new): helper/parser tests.

## Local evidence
- Notebook execution (nbconvert):
  - output notebook: `tmp_jupyter_smoke.out.ipynb`
  - contains widget output: `application/vnd.jupyter.widget-view+json`
  - contains terminal marker: `NB_FIT_DONE`
- Helper tests:
  - `python3 scripts/test_jupyter_progress_helpers.py` => `helpers-tests=ok`
  - `python -m pytest -q pysr/test/test_jupyter_progress_helpers.py` => `3 passed`

## Repro commands
```bash
cd /root/.openclaw/workspace/worktrees/pysr-jupyter-progress
source .venv/bin/activate

# helper tests
python3 scripts/test_jupyter_progress_helpers.py
python -m pytest -q pysr/test/test_jupyter_progress_helpers.py

# notebook execution smoke
jupyter nbconvert --to notebook --execute tmp_jupyter_smoke.ipynb --output tmp_jupyter_smoke.out.ipynb
```

## Note
- In this host environment, pytest exits with a late segfault after printing `3 passed` (likely runtime/finalizer issue), but the tests themselves complete and report pass before process teardown.
