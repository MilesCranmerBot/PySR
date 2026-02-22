"""Jupyter progress with file-based monitoring.

This module is meant for Jupyter/Colab where widget updates must occur on the
main Python thread.

Approach:
- Julia writes progress JSON to a file (via SymbolicRegression.jl ProgressFileWriter)
- Python runs the Julia call in a background thread
- The main thread polls the file and updates an ipywidgets progress display
"""

from __future__ import annotations

import json
import os
import threading
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Union


class _IpywidgetsProgressDisplay:
    """Simple ipywidgets-based progress display."""

    def __init__(self, total: int):
        from IPython.display import display
        from ipywidgets import HTML, IntProgress, VBox

        self._total = int(total)
        self._current = 0
        self._start_time = time.time()

        self._bar = IntProgress(
            value=0,
            min=0,
            max=max(self._total, 1),
            description="PySR",
            style={"description_width": "initial"},
        )
        self._label = HTML(value=f"0 / {self._total}")
        self._widget = VBox([self._bar, self._label])
        display(self._widget)

    def update(self, current: int, total: int) -> None:
        """Update progress."""
        current = int(current)
        total = int(total)
        self._current = current
        if total != self._total:
            self._total = total
            self._bar.max = max(total, 1)
        self._bar.value = min(max(current, 0), self._bar.max)
        elapsed = time.time() - self._start_time
        self._label.value = f"{current} / {total} ({elapsed:.1f}s)"

    def set_message(self, message: str) -> None:
        """Set status message."""
        elapsed = time.time() - self._start_time
        self._label.value = f"{message} ({elapsed:.1f}s)"

    def close(self) -> None:
        """Close widget."""
        try:
            self._widget.close()
        except Exception:
            pass


class FileBasedProgressExecutor:
    """Execute a function in a background thread while polling a progress file.

    The *widget updates* are performed on the caller thread (intended to be the
    main thread in a notebook).
    """

    def __init__(self):
        self._result: Optional[Any] = None
        self._exception: Optional[BaseException] = None
        self._done = threading.Event()

    def _run_julia(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            self._result = func(*args, **kwargs)
        except BaseException as e:
            self._exception = e
        finally:
            self._done.set()

    def execute(self, func: Callable[..., Any], widget: _IpywidgetsProgressDisplay,
                progress_file: Union[str, os.PathLike[str], None], total_iters: int,
                *args: Any, poll_interval_s: float = 0.2, **kwargs: Any) -> Any:
        """Execute `func` with file-based progress.

        Args:
            func: Callable to run (e.g., SymbolicRegression.equation_search)
            widget: Progress display to update
            progress_file: Path to JSON progress file. If None, a temp file is created.
            total_iters: Expected total iterations (used as a fallback)
            poll_interval_s: Main-thread polling interval
            *args/**kwargs: passed to func; we also pass `progress_file=<path>`

        Returns:
            func result
        """
        total_iters = int(total_iters)

        created_tmp = False
        if progress_file is None:
            fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="pysr_progress_")
            os.close(fd)
            progress_path = Path(tmp_path)
            created_tmp = True
        else:
            progress_path = Path(progress_file)

        # Ensure the file exists with a valid initial payload.
        try:
            progress_path.write_text(json.dumps({"current": 0, "total": total_iters}))
        except Exception:
            # Best effort; polling loop will tolerate missing/bad file.
            pass

        # Start background execution.
        self._done.clear()
        self._result = None
        self._exception = None

        thread = threading.Thread(
            target=self._run_julia,
            args=(func,) + args,
            kwargs={**kwargs, "progress_file": str(progress_path)},
            daemon=False,
        )
        thread.start()

        last_current: Optional[int] = None
        last_total: Optional[int] = None

        try:
            # Poll from *this* thread (intended: main thread).
            while thread.is_alive() and not self._done.is_set():
                try:
                    if progress_path.exists():
                        data = json.loads(progress_path.read_text())
                        current = int(data.get("current", 0))
                        total = int(data.get("total", total_iters))
                        if current != last_current or total != last_total:
                            last_current, last_total = current, total
                            widget.update(current, total)
                except Exception:
                    pass
                time.sleep(poll_interval_s)

            thread.join()

            # Final update.
            try:
                if progress_path.exists():
                    data = json.loads(progress_path.read_text())
                    current = int(data.get("current", total_iters))
                    total = int(data.get("total", total_iters))
                    widget.update(current, total)
                else:
                    widget.update(total_iters, total_iters)
            except Exception:
                pass

            if self._exception is not None:
                raise self._exception

            return self._result
        finally:
            if created_tmp:
                try:
                    progress_path.unlink()
                except Exception:
                    pass
