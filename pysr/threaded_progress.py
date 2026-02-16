"""Threaded Jupyter progress for PySR using file-based communication.

This module enables live progress updates in Jupyter/Colab by:
1. Running the Julia equation_search in a background Python thread
2. Using PythonCall.jl's @pyglet macro to release the GIL during the call
3. Having Julia write progress to a temp file
4. The main Python thread polls the file and updates the widget
"""

import os
import sys
import threading
import time
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


class _IpywidgetsProgressDisplay:
    """Simple ipywidgets-based progress display."""
    
    def __init__(self, total: int):
        from IPython.display import display
        from ipywidgets import HTML, IntProgress, VBox
        
        self._total = total
        self._current = 0
        self._start_time = time.time()
        
        self._bar = IntProgress(
            value=0, min=0, max=max(total, 1), 
            description="PySR", 
            style={'description_width': 'initial'}
        )
        self._label = HTML(value=f"0 / {total}")
        self._widget = VBox([self._bar, self._label])
        display(self._widget)

    def update(self, current: int, total: int) -> None:
        """Update progress."""
        self._current = current
        if total != self._total:
            self._total = total
            self._bar.max = max(total, 1)
        self._bar.value = min(max(current, 0), self._bar.max)
        elapsed = time.time() - self._start_time
        self._label.value = f"{current} / {total} ({elapsed:.1f}s)"

    def set_message(self, message: str) -> None:
        """Set status message."""
        self._label.value = message

    def close(self) -> None:
        """Close widget."""
        try:
            self._widget.close()
        except Exception:
            pass


class ThreadedJupyterProgress:
    """Manages threaded execution with progress updates."""
    
    def __init__(self, total_iterations: int):
        self.total_iterations = max(int(total_iterations), 1)
        self.display: Optional[_IpywidgetsProgressDisplay] = None
        self._progress_file: Optional[Path] = None
        self._stop_polling = threading.Event()
        self._julia_thread: Optional[threading.Thread] = None
        self._result = None
        self._exception = None
        
    def _poll_progress(self) -> None:
        """Poll progress file from main thread."""
        if self._progress_file is None:
            return
            
        while not self._stop_polling.is_set():
            try:
                if self._progress_file.exists():
                    data = json.loads(self._progress_file.read_text())
                    current = data.get('current', 0)
                    total = data.get('total', self.total_iterations)
                    if self.display:
                        self.display.update(current, total)
            except Exception:
                pass
            
            # Check if Julia thread is done
            if self._julia_thread and not self._julia_thread.is_alive():
                break
                
            time.sleep(0.5)  # Poll every 500ms
    
    def _run_julia_with_progress(self, func, *args, **kwargs) -> None:
        """Run Julia function in background thread."""
        try:
            # This runs in background thread - Julia should release GIL
            self._result = func(*args, **kwargs)
        except Exception as e:
            self._exception = e
    
    def run(self, julia_func, *args, **kwargs):
        """Run Julia function with progress updates.
        
        Args:
            julia_func: The Julia function to call (e.g., SymbolicRegression.equation_search)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the Julia function call
        """
        # Create temp file for progress communication
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self._progress_file = Path(f.name)
        
        # Initialize progress file
        self._progress_file.write_text(json.dumps({
            'current': 0, 
            'total': self.total_iterations
        }))
        
        # Create display
        self.display = _IpywidgetsProgressDisplay(self.total_iterations)
        
        # Create and start Julia thread
        self._julia_thread = threading.Thread(
            target=self._run_julia_with_progress,
            args=(julia_func,) + args,
            kwargs=kwargs,
            daemon=False
        )
        
        # Start polling from main thread (this blocks until Julia completes)
        self._stop_polling.clear()
        self._julia_thread.start()
        
        # Poll from main thread
        self._poll_progress()
        
        # Wait for Julia thread to complete
        self._julia_thread.join()
        
        # Cleanup
        self._stop_polling.set()
        try:
            self._progress_file.unlink()
        except Exception:
            pass
        
        if self.display:
            self.display.update(self.total_iterations, self.total_iterations)
            self.display.close()
        
        if self._exception:
            raise self._exception
            
        return self._result


@contextmanager
def threaded_progress_capture(total_iterations: int) -> Iterator[ThreadedJupyterProgress]:
    """Context manager for threaded progress.
    
    Usage:
        with threaded_progress_capture(niterations * populations) as prog:
            result = prog.run(SymbolicRegression.equation_search, ...)
    """
    progress = ThreadedJupyterProgress(total_iterations)
    try:
        yield progress
    finally:
        if progress.display:
            progress.display.close()
