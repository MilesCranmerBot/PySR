"""Jupyter-compatible progress display helpers for PySR."""

from __future__ import annotations

import os
import re
import sys
import threading
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
        pass
    def close(self) -> None:
        pass


class _IpywidgetsProgressDisplay:
    """Widget display using ipywidgets - more reliable than tqdm in notebooks."""
    
    def __init__(self, total: int):
        from IPython.display import display
        from ipywidgets import HTML, IntProgress, VBox
        import ipywidgets as widgets

        self._total = total
        self._current = 0
        
        # Create widget with explicit layout
        self._bar = IntProgress(
            value=0, 
            min=0, 
            max=max(total, 1), 
            description="PySR",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )
        self._label = HTML(value=f"0 / {total}")
        self._widget = VBox([self._bar, self._label])
        
        # Display and keep reference
        display(self._widget)

    def update(self, current: int, total: int) -> None:
        """Update widget values."""
        import ipywidgets as widgets
        
        self._current = current
        if total != self._total:
            self._total = total
            self._bar.max = max(total, 1)
        
        # Direct value assignment
        self._bar.value = min(max(current, 0), self._bar.max)
        self._label.value = f"{current} / {total}"

    def close(self) -> None:
        """Close the widget."""
        try:
            self._bar.close()
            self._label.close()
            self._widget.close()
        except Exception:
            pass


def _is_notebook_session() -> bool:
    """Detect if we're running in a notebook."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        return ipython.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _create_display(total: int) -> _ProgressDisplay:
    """Create progress display - prefer ipywidgets for reliability."""
    try:
        # Only try ipywidgets, skip tqdm due to threading issues
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


class _ProgressCaptureStream:
    """Captures stdout/stderr and parses progress lines."""
    
    def __init__(self, target_stream, parser: _ProgressLineParser):
        self._target = target_stream
        self._parser = parser
        self._buffer = ""
        self._lock = threading.Lock()

    def _drain_complete_lines(self) -> None:
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


class JupyterProgressContext:
    """Context manager for Jupyter progress display."""

    def __init__(self, total_iterations: int):
        self.total_iterations = max(int(total_iterations), 1)
        self.display: _ProgressDisplay = _NullProgressDisplay()
        self._parser = _ProgressLineParser(self._on_progress)
        self._current = 0
        self._lock = threading.Lock()

    def _on_progress(self, current: int, total: int) -> None:
        """Called when progress is detected (may be from any thread)."""
        with self._lock:
            self._current = current
        # Update display immediately - ipywidgets should handle thread safety
        try:
            self.display.update(current, total)
        except Exception:
            pass

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Capture stdout/stderr and display progress."""
        self.display = _create_display(self.total_iterations)
        self.display.update(0, self.total_iterations)
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_capture = _ProgressCaptureStream(old_stdout, self._parser)
        stderr_capture = _ProgressCaptureStream(old_stderr, self._parser)
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield
        finally:
            stdout_capture.flush()
            stderr_capture.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
            # Final update
            with self._lock:
                final = self._current
            try:
                self.display.update(self.total_iterations, self.total_iterations)
            except Exception:
                pass
            self.display.close()


def should_use_jupyter_progress(*, progress: bool, verbosity: int, is_single_output: bool) -> bool:
    """Whether PySR should use Python-side notebook progress handling."""
    if not progress or verbosity <= 0 or not is_single_output:
        return False
    disable_progress = os.environ.get("PYSR_DISABLE_JUPYTER_PROGRESS", "").lower()
    if disable_progress in {"1", "true", "yes", "on"}:
        return False
    return _is_notebook_session()
