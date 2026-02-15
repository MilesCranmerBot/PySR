"""Prototype B: Julia-side progress emission bridged to Python output parsing."""

from __future__ import annotations

import re
import subprocess


JULIA_BIN = "/root/.julia/juliaup/julia-1.10.10+0.x64.linux.gnu/bin/julia"
PROGRESS_RE = re.compile(r"^PYSR_PROGRESS\s+(\d+)\s*/\s*(\d+)$")


def run() -> None:
    julia_program = r"""
for i in 1:8
    println("PYSR_PROGRESS $(i) / 8")
    sleep(0.01)
end
"""
    proc = subprocess.run(
        [JULIA_BIN, "-e", julia_program],
        capture_output=True,
        text=True,
        check=True,
    )
    updates: list[tuple[int, int]] = []
    for line in proc.stdout.splitlines():
        match = PROGRESS_RE.match(line.strip())
        if match:
            updates.append((int(match.group(1)), int(match.group(2))))

    print("prototype=B")
    print("updates=", updates)


if __name__ == "__main__":
    run()
