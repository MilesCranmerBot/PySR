"""Jupyter progress with file-based monitoring.

Uses Julia's ProgressFileWriter to write progress to a file,
which Python monitors from the main thread.
"""
import json
import os
import threading
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional


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
        elapsed = time.time() - self._start_time
        self._label.value = f"{message} ({elapsed:.1f}s)"

    def close(self) -> None:
        """Close widget."""
        try:
            self._widget.close()
        except Exception:
            pass


class JupyterProgressMonitor:
    """Monitors progress file and updates widget from main thread."""
    
    def __init__(self, progress_file: Path, widget: _IpywidgetsProgressDisplay, total: int):
        self.progress_file = progress_file
        self.widget = widget
        self.total = total
        self._stop = threading.Event()
        self._current = 0
        
    def monitor(self) -> None:
        """Monitor progress file until completion.
        
        This runs in the main thread, updating the widget periodically.
        """
        while not self._stop.is_set():
            try:
                if self.progress_file.exists():
                    data = json.loads(self.progress_file.read_text())
                    current = data.get('current', 0)
                    if current != self._current:
                        self._current = current
                        self.widget.update(current, self.total)
            except Exception:
                pass
            
            # Short sleep to not block completely
            time.sleep(0.2)
    
    def stop(self) -> None:
        self._stop.set()


class FileBasedProgressExecutor:
    """Executes Julia function with file-based progress tracking."""
    
    def __init__(self):
        self._result: Optional[Any] = None
        self._exception: Optional[Exception] = None
        self._done = threading.Event()
        
    def _run_julia(self, func: Callable, progress_file: str, *args, **kwargs) -> None:
        """Run Julia function in background thread."""
        try:
            # Pass progress_file to Julia function
            self._result = func(*args, progress_file=progress_file, **kwargs)
        except Exception as e:
            self._exception = e
        finally:
            self._done.set()
    
    def execute(self, func: Callable, widget: _IpywidgetsProgressDisplay, 
                total_iters: int, *args, **kwargs) -> Any:
        """Execute with progress monitoring via file.
        
        Args:
            func: Julia function to call
            widget: Widget to update
            total_iters: Total iterations expected
            *args, **kwargs: Args for func
        """
        # Create progress file
        fd, progress_path = tempfile.mkstemp(suffix='.json', prefix='pysr_progress_')
        os.close(fd)
        progress_file = Path(progress_path)
        progress_file.write_text(json.dumps({'current': 0, 'total': total_iters}))
        
        try:
            # Start Julia in background thread
            julia_thread = threading.Thread(
                target=self._run_julia,
                args=(func, str(progress_file)) + args,
                kwargs=kwargs,
                daemon=False
            )
            
            self._done.clear()
            julia_thread.start()
            
            # Monitor from main thread
            monitor = JupyterProgressMonitor(progress_file, widget, total_iters)
            
            # Run monitoring loop
            while not self._done.is_set():
                monitor.monitor()
                if self._done.is_set():
                    break
                time.sleep(0.1)
            
            # Wait for Julia to complete
            julia_thread.join()
            
            # Final update
            try:
                if progress_file.exists():
                    data = json.loads(progress_file.read_text())
                    widget.update(data.get('current', total_iters), total_iters)
            except Exception:
                pass
            
            if self._exception:
                raise self._exception
                
            return self._result
            
        finally:
            # Cleanup
            monitor.stop()
            try:
                progress_file.unlink()
            except Exception:
                pass
