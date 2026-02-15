"""Prototype C: progress UI driven by polling a JSON progress file."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path


def _writer(path: Path, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for current in range(1, total + 1):
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"current": current, "total": total}))
        tmp_path.replace(path)
        time.sleep(0.01)


def run() -> None:
    total = 12
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / "progress.json"
        thread = threading.Thread(target=_writer, args=(progress_file, total), daemon=True)
        thread.start()

        updates: list[tuple[int, int]] = []
        last_current = -1
        while thread.is_alive() or progress_file.exists():
            if progress_file.exists():
                try:
                    payload = json.loads(progress_file.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.005)
                    continue
                current = int(payload["current"])
                total_from_file = int(payload["total"])
                if current != last_current:
                    updates.append((current, total_from_file))
                    last_current = current
                if current >= total_from_file and not thread.is_alive():
                    break
            time.sleep(0.005)

    print("prototype=C")
    print("updates=", updates)


if __name__ == "__main__":
    run()
