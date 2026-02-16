"""Threaded execution for PySR Jupyter progress.

Runs equation_search in a background thread so the main thread can update widgets.
"""
import threading
import time
from typing import Any, Callable, Optional


class ThreadedExecutor:
    """Executes a function in a background thread with progress updates."""
    
    def __init__(self):
        self._result: Optional[Any] = None
        self._exception: Optional[Exception] = None
        self._done = threading.Event()
        
    def _run_in_thread(self, func: Callable, *args, **kwargs) -> None:
        """Run function in background thread."""
        try:
            self._result = func(*args, **kwargs)
        except Exception as e:
            self._exception = e
        finally:
            self._done.set()
    
    def execute(self, func: Callable, progress_widget: Any, *args, **kwargs) -> Any:
        """Execute function with live progress updates.
        
        Args:
            func: The function to execute (e.g., equation_search)
            progress_widget: Widget to update with progress
            *args, **kwargs: Arguments for func
            
        Returns:
            The result of func
        """
        # Start function in background thread
        # PythonCall.jl releases the GIL, so this works
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(func,) + args,
            kwargs=kwargs,
            daemon=False
        )
        
        self._done.clear()
        thread.start()
        
        # Poll from main thread (updates widget)
        iteration = 0
        while not self._done.is_set():
            # Update widget with pulsing indicator
            if progress_widget:
                dots = "." * (iteration % 4)
                try:
                    progress_widget.set_message(f"Running{dots}")
                except Exception:
                    pass
            iteration += 1
            time.sleep(0.5)
        
        # Wait for completion
        thread.join()
        
        if self._exception:
            raise self._exception
            
        return self._result
