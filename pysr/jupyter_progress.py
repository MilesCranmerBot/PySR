"""Jupyter progress widget for PySR."""
import time


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
