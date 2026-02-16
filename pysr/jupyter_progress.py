"""Jupyter-compatible progress display helpers for PySR."""

from __future__ import annotations

import os
import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Protocol, Tuple


_PROGRESS_PATTERN = re.compile(r"Progress:\s*(\d+)\s*/\s*(\d+)\s*total iterations")
_EVOLVING_PATTERN = re.compile(r"Evolving for\s*(\d+)\s*iterations\.\.\.\s*(\d+)%\|")


class _ProgressDisplay(Protocol):
    def update(self, current: int, total: int) -> None: ...

    def close(self) -> None: ...


class _NullProgressDisplay:
    def update(self, current: int, total: int) -> None:
        return None

    def close(self) -> None:
        return None


class _TqdmProgressDisplay:
    def __init__(self, total: int):
        from tqdm.notebook import tqdm

        self._bar = tqdm(total=total, desc="PySR fit", leave=True)
        self._current = 0

    def update(self, current: int, total: int) -> None:
        if total != self._bar.total:
            self._bar.total = total
        delta = max(0, current - self._current)
        if delta > 0:
            self._bar.update(delta)
        self._current = current

    def close(self) -> None:
        self._bar.close()


class _IpywidgetsProgressDisplay:
    def __init__(self, total: int):
        from IPython.display import display
        from ipywidgets import HTML, IntProgress, VBox

        self._bar = IntProgress(value=0, min=0, max=max(total, 1), description="PySR fit")
        self._label = HTML(value=f"0 / {total} iterations")
        self._widget = VBox([self._bar, self._label])
        display(self._widget)

    def update(self, current: int, total: int) -> None:
        self._bar.max = max(total, 1)
        self._bar.value = min(max(current, 0), self._bar.max)
        self._label.value = f"{current} / {total} iterations"

    def close(self) -> None:
        return None


def _is_notebook_session() -> bool:
    # Colab does not always expose the same IPython shell metadata as classic Jupyter.
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return True

    try:
        from IPython import get_ipython
    except Exception:
        return False

    ipython = get_ipython()
    if ipython is None:
        return False
    return ipython.__class__.__name__ == "ZMQInteractiveShell"


def _get_iopub_thread():
    """Best-effort access to ipykernel's IOPubThread (for thread-safe widget updates)."""
    try:
        from ipykernel.ipkernel import IPythonKernel  # type: ignore

        ip = IPythonKernel.instance()
        return getattr(ip, "iopub_thread", None)
    except Exception:
        return None


def _create_display(total: int) -> _ProgressDisplay:
    try:
        return _TqdmProgressDisplay(total=total)
    except Exception:
        pass

    try:
        return _IpywidgetsProgressDisplay(total=total)
    except Exception:
        return _NullProgressDisplay()


@dataclass
class _ProgressLineParser:
    on_progress: Callable[[int, int], None]

    def parse_line(self, line: str) -> None:
        progress_matches = list(_PROGRESS_PATTERN.finditer(line))
        if progress_matches:
            match = progress_matches[-1]
            current = int(match.group(1))
            total = int(match.group(2))
            self.on_progress(current, total)
            return

        evolving_matches = list(_EVOLVING_PATTERN.finditer(line))
        if evolving_matches:
            match = evolving_matches[-1]
            total = int(match.group(1))
            pct = int(match.group(2))
            current = int(round(total * pct / 100.0))
            self.on_progress(current, total)


@dataclass
class _StreamProxy:
    _stream: object
    write: Callable[[str], int]
    flush: Callable[[], None] | None = None

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


class _ProgressCaptureStream:
    def __init__(self, target_stream, parser: _ProgressLineParser):
        self._target = target_stream
        self._parser = parser
        self._buffer = ""
        self._lock = threading.RLock()

    def _drain_complete_lines(self) -> None:
        # ProgressMeter often updates in-place with carriage returns (`\r`) rather
        # than newline-terminated lines. Treat both as parse boundaries.
        while True:
            newline_idx = self._buffer.find("\n")
            carriage_idx = self._buffer.find("\r")
            candidates = [idx for idx in (newline_idx, carriage_idx) if idx != -1]
            if not candidates:
                break
            split_idx = min(candidates)
            line = self._buffer[:split_idx]
            self._buffer = self._buffer[split_idx + 1 :]
            self._parser.parse_line(line)

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            written = self._target.write(text)
            self._buffer += text
            self._drain_complete_lines()
            # Also parse the in-flight buffer in case progress updates arrive
            # without line delimiters (seen in some notebook frontends).
            if self._buffer:
                self._parser.parse_line(self._buffer)
            return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._parser.parse_line(self._buffer)
                self._buffer = ""
            if hasattr(self._target, "flush"):
                self._target.flush()

    def __getattr__(self, name: str):
        return getattr(self._target, name)


class JupyterProgressContext:
    """Capture text progress lines and render a notebook progress widget."""

    def __init__(self, total_iterations: int):
        self.total_iterations = max(int(total_iterations), 1)
        self.display: _ProgressDisplay = _NullProgressDisplay()
        self._parser = _ProgressLineParser(self._on_progress)
        self._current = 0

        # Progress updates often come from non-main threads (e.g., ipykernel watchfd thread,
        # or Julia threads calling into Python). Some notebook frontends (notably Colab)
        # are much more reliable if widget updates are sent from ipykernel's IOPub thread.
        self._iopub_thread = None
        self._update_lock = threading.Lock()
        self._pending_update: Optional[Tuple[int, int]] = None
        self._update_scheduled = False
        self._active = False

    def _on_progress(self, current: int, total: int) -> None:
        self._current = current
        self._queue_update(current, total)

    def _queue_update(self, current: int, total: int) -> None:
        # Coalesce frequent updates to avoid flooding the frontend with widget state
        # messages. This also lets us route the actual widget update through the
        # ipykernel IOPub thread when available.
        with self._update_lock:
            if not self._active:
                return
            self._pending_update = (current, total)
            if self._update_scheduled:
                return
            self._update_scheduled = True

        iopub_thread = self._iopub_thread
        if iopub_thread is not None:
            iopub_thread.schedule(self._apply_pending_update)
        else:
            self._apply_pending_update()

    def _apply_pending_update(self) -> None:
        # Runs either on ipykernel's IOPub thread or the current thread.
        # Process all pending updates but limit iterations to prevent infinite loops.
        max_iterations = 100
        for _ in range(max_iterations):
            with self._update_lock:
                if not self._active:
                    self._pending_update = None
                    self._update_scheduled = False
                    return
                pending = self._pending_update
                self._pending_update = None
                if pending is None:
                    self._update_scheduled = False
                    return

            current, total = pending
            try:
                self.display.update(current, total)
            except Exception:
                # Never let UI plumbing crash a model fit.
                pass

    @contextmanager
    def capture(self) -> Iterator[None]:
        self.display = _create_display(self.total_iterations)
        self.display.update(0, self.total_iterations)

        # Mark active *before* any output starts flowing.
        self._iopub_thread = _get_iopub_thread()
        with self._update_lock:
            self._active = True
            self._pending_update = None
            self._update_scheduled = False

        old_stdout, old_stderr = sys.stdout, sys.stderr

        stdout_old_write = old_stdout.write
        stdout_old_flush = getattr(old_stdout, "flush", None)
        stderr_old_write = old_stderr.write
        stderr_old_flush = getattr(old_stderr, "flush", None)

        stdout_proxy = _StreamProxy(old_stdout, stdout_old_write, stdout_old_flush)
        stderr_proxy = _StreamProxy(old_stderr, stderr_old_write, stderr_old_flush)
        stdout_capture_stream = _ProgressCaptureStream(stdout_proxy, self._parser)
        stderr_capture_stream = _ProgressCaptureStream(stderr_proxy, self._parser)

        try:
            old_stdout.write = stdout_capture_stream.write
            if stdout_old_flush is not None:
                old_stdout.flush = stdout_capture_stream.flush

            old_stderr.write = stderr_capture_stream.write
            if stderr_old_flush is not None:
                old_stderr.flush = stderr_capture_stream.flush

            yield
        finally:
            stdout_capture_stream.flush()
            stderr_capture_stream.flush()

            old_stdout.write = stdout_old_write
            if stdout_old_flush is not None:
                old_stdout.flush = stdout_old_flush

            old_stderr.write = stderr_old_write
            if stderr_old_flush is not None:
                old_stderr.flush = stderr_old_flush

            # Prevent any late/asynchronous progress callbacks from trying to update
            # a closed widget.
            with self._update_lock:
                self._active = False
                self._pending_update = None
                self._update_scheduled = False

            self.display.update(self.total_iterations, self.total_iterations)
            self.display.close()


def should_use_jupyter_progress(*, progress: bool, verbosity: int, is_single_output: bool) -> bool:
    """Whether PySR should use Python-side notebook progress handling."""
    if not progress or verbosity <= 0 or not is_single_output:
        return False

    disable_progress = os.environ.get("PYSR_DISABLE_JUPYTER_PROGRESS", "").lower()
    if disable_progress in {"1", "true", "yes", "on"}:
        return False

    return _is_notebook_session()
