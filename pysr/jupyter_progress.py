"""Jupyter-compatible progress display helpers for PySR."""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol


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
        match = _PROGRESS_PATTERN.search(line)
        if match is not None:
            current = int(match.group(1))
            total = int(match.group(2))
            self.on_progress(current, total)
            return

        match = _EVOLVING_PATTERN.search(line)
        if match is not None:
            total = int(match.group(1))
            pct = int(match.group(2))
            current = int(round(total * pct / 100.0))
            self.on_progress(current, total)


class _ProgressCaptureStream:
    def __init__(self, target_stream, parser: _ProgressLineParser):
        self._target = target_stream
        self._parser = parser
        self._buffer = ""

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
        written = self._target.write(text)
        self._buffer += text
        self._drain_complete_lines()
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
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

    def _on_progress(self, current: int, total: int) -> None:
        self._current = current
        self.display.update(current, total)

    @contextmanager
    def capture(self) -> Iterator[None]:
        self.display = _create_display(self.total_iterations)
        self.display.update(0, self.total_iterations)
        stdout_capture = _ProgressCaptureStream(sys.stdout, self._parser)
        stderr_capture = _ProgressCaptureStream(sys.stderr, self._parser)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield
        finally:
            stdout_capture.flush()
            stderr_capture.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.display.update(self.total_iterations, self.total_iterations)
            self.display.close()


def should_use_jupyter_progress(*, progress: bool, verbosity: int, is_single_output: bool) -> bool:
    """Whether PySR should use Python-side notebook progress handling."""
    if not progress or verbosity <= 0 or not is_single_output:
        return False
    return _is_notebook_session()
