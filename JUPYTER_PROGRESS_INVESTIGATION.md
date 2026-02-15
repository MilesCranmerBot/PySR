# Jupyter Progress Investigation (PySR `fit()`)

## Scope
- Focused on Jupyter notebook progress UX for `fit()`.
- Explicitly did not modify `input_stream`, `devnull`, or input-stream auto-selection logic.

## 1) Current progress path (Julia -> Python)
- `pysr/sr.py` passes `progress=...` to `SymbolicRegression.equation_search(...)`.
- Existing behavior had a Python guard in `_mutate_parameter` that disabled progress when `sys.stdout` lacked `buffer` (Jupyter heuristic).
- In SymbolicRegression.jl:
  - `progress=true` uses Julia progress bar (`update_progress_bar!`).
  - `progress=false` and `verbosity>0` prints textual status including:
    - `Progress: X / Y total iterations (...)`

Command evidence:
```bash
rg -n "progress|equation_search\\(" pysr/sr.py -S
```
```text
2295:            progress=runtime_params.progress ...
```
```bash
sed -n '1000,1115p' ~/.julia/packages/SymbolicRegression/VApOk/src/SymbolicRegression.jl
```
```text
if ropt.verbosity > 0 && !ropt.progress ...
    print_search_state(...)
```

## 2) Alternatives implemented + tested

### A) Python-side notebook UI driven by parsed progress signals
- Prototype file: `scripts/jupyter_progress_prototype_a_python_ui.py`
- Mechanism:
  - Parse text lines matching `Progress: X / Y total iterations`.
  - Update a Python-side display state.
- Command:
```bash
python3 scripts/jupyter_progress_prototype_a_python_ui.py
```
- Output:
```text
prototype=A
updates= [(3, 10), (10, 10)]
```
- Result: Works reliably with split/chunked stream writes.

### B) Julia-side progress emission bridged into Python output parsing
- Prototype file: `scripts/jupyter_progress_prototype_b_julia_output_bridge.py`
- Mechanism:
  - Julia emits machine-parseable progress lines.
  - Python parses bridge output and reconstructs progress.
- Command:
```bash
python3 scripts/jupyter_progress_prototype_b_julia_output_bridge.py
```
- Output:
```text
prototype=B
updates= [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)]
```
- Result: Strong signal quality; requires explicit Julia-side emission contract.

### C) File-polling progress transport
- Prototype file: `scripts/jupyter_progress_prototype_c_file_poll.py`
- Mechanism:
  - Producer writes JSON progress snapshots.
  - Consumer polls file and updates UI.
- Command:
```bash
python3 scripts/jupyter_progress_prototype_c_file_poll.py
```
- Output:
```text
prototype=C
updates= [(1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12)]
```
- Result: Functional but introduces file I/O overhead and synchronization complexity.

## 3) Selected design
- Chosen: **A (Python-side notebook UI + parsed progress lines)**.
- Why:
  - No backend protocol changes required.
  - Preserves existing terminal behavior.
  - Works with optional notebook UI dependencies (`tqdm.notebook`, then `ipywidgets`, then no-op fallback).
- Implemented in:
  - `pysr/jupyter_progress.py`
  - `pysr/sr.py`

Design details:
- Detect notebook sessions (`ZMQInteractiveShell`).
- In notebook + eligible progress case:
  - Disable Julia bar for that call (`progress=False` for `equation_search`) to force textual progress lines.
  - Wrap `sys.stdout` and `sys.stderr`.
  - Parse `Progress: X / Y total iterations` lines.
  - Render notebook progress via:
    1. `tqdm.notebook`
    2. `ipywidgets` (`IntProgress`)
    3. no-op fallback
- Outside notebook: unchanged behavior (Julia progress/CLI remains as before).

## 4) Tests and validation

### Added tests
- `pysr/test/test_jupyter_progress_helpers.py`
- Updated:
  - `pysr/test/test_main.py`
    - `test_progress_disabled_when_stdout_lacks_buffer` ->
      `test_progress_not_disabled_when_stdout_lacks_buffer`
- Also added standalone script tests:
  - `scripts/test_jupyter_progress_helpers.py`

### Commands run
```bash
python3 scripts/test_jupyter_progress_helpers.py
```
```text
helpers-tests=ok
```

```bash
python3 -m compileall pysr/jupyter_progress.py pysr/sr.py scripts/jupyter_progress_prototype_a_python_ui.py scripts/jupyter_progress_prototype_b_julia_output_bridge.py scripts/jupyter_progress_prototype_c_file_poll.py
```
```text
Compiling 'pysr/sr.py'...
Compiling 'scripts/jupyter_progress_prototype_a_python_ui.py'...
Compiling 'scripts/jupyter_progress_prototype_b_julia_output_bridge.py'...
Compiling 'scripts/jupyter_progress_prototype_c_file_poll.py'...
```

## 5) Regression expectation
- Terminal/non-notebook progress path remains unchanged.
- Notebook path now uses Python-side UI instead of force-disabling progress.

## 6) Repro demo
- Lightweight parser demo:
```bash
python3 scripts/jupyter_progress_prototype_a_python_ui.py
```
- Full PySR notebook demo command (in a real env with `juliacall` and notebook kernel installed):
```python
from pysr import PySRRegressor
import numpy as np

X = np.random.randn(200, 3)
y = X[:, 0] - X[:, 1]
model = PySRRegressor(niterations=40, populations=8, progress=True, verbosity=1)
model.fit(X, y)
```
