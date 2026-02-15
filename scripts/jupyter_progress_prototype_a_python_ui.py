"""Prototype A: Python-side progress UI from parsed Julia text signals."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass


@dataclass
class RecorderDisplay:
    updates: list[tuple[int, int]]

    def update(self, current: int, total: int) -> None:
        self.updates.append((current, total))

    def close(self) -> None:
        return None


def run() -> None:
    from pathlib import Path
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "pysr" / "jupyter_progress.py"
    spec = importlib.util.spec_from_file_location("_jp", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    updates: list[tuple[int, int]] = []
    display = RecorderDisplay(updates)
    parser = module._ProgressLineParser(display.update)
    stream = module._ProgressCaptureStream(io.StringIO(), parser)

    stream.write("Expressions evaluated per second: 1.2e3\n")
    stream.write("Progress: 3 / 10 total ")
    stream.write("iterations (30.0%)\n")
    stream.write("Progress: 10 / 10 total iterations (100.0%)\n")
    stream.flush()
    display.close()

    print("prototype=A")
    print("updates=", updates)


if __name__ == "__main__":
    run()
