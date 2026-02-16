"""Progress file monitoring for PySR Jupyter integration.

This module enables live progress updates by having Julia write to a file
and Python reading from it in the main thread.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional


class ProgressFileMonitor:
    """Monitors a progress file written by Julia and updates a widget."""
    
    def __init__(self, progress_file: Path, widget, total_iterations: int):
        self.progress_file = progress_file
        self.widget = widget
        self.total_iterations = total_iterations
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._current = 0
        
    def start(self) -> None:
        """Start monitoring the progress file."""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self) -> None:
        """Background thread that reads progress file."""
        while not self._stop_event.is_set():
            try:
                if self.progress_file.exists():
                    data = json.loads(self.progress_file.read_text())
                    current = data.get('current', 0)
                    total = data.get('total', self.total_iterations)
                    if current != self._current:
                        self._current = current
                        # Schedule update on main thread via IPython
                        self._schedule_update(current, total)
            except Exception:
                pass
            time.sleep(0.5)  # Poll every 500ms
    
    def _schedule_update(self, current: int, total: int) -> None:
        """Schedule widget update on main thread."""
        try:
            # Try to use IPython's event loop
            from IPython import get_ipython
            ip = get_ipython()
            if ip and hasattr(ip, 'kernel'):
                # In IPython kernel, use asyncio
                import asyncio
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(lambda: self.widget.update(current, total))
            else:
                # Direct update (may not work from background thread)
                self.widget.update(current, total)
        except Exception:
            # Fallback: just try direct update
            try:
                self.widget.update(current, total)
            except Exception:
                pass


def create_progress_file() -> Path:
    """Create a temporary progress file and return its path."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.json', prefix='pysr_progress_')
    os.close(fd)
    progress_file = Path(path)
    # Initialize with zeros
    progress_file.write_text(json.dumps({'current': 0, 'total': 100}))
    return progress_file
