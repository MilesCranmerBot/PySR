"""Kernel-level test for *live* widget updates.

This test is optional: it requires ipykernel, jupyter_client, and ipywidgets.

Goal: verify that updating an ipywidgets progress bar during execution produces
multiple comm messages on IOPub *before* the kernel becomes idle.

We load the file-based executor module by filename to avoid importing pysr/__init__
(which can require juliacall at import time).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path


class TestWidgetCommLiveKernel(unittest.TestCase):
    def test_widget_comm_msgs_arrive_before_idle(self):
        try:
            import jupyter_client  # type: ignore
            from jupyter_client import KernelManager  # type: ignore
        except Exception:
            self.skipTest("requires jupyter_client")

        try:
            import ipykernel  # noqa: F401
        except Exception:
            self.skipTest("requires ipykernel")

        try:
            import ipywidgets  # noqa: F401
        except Exception:
            self.skipTest("requires ipywidgets")

        # Build code to run inside the kernel.
        module_path = (Path(__file__).resolve().parents[1] / "jupyter_progress.py").as_posix()
        code = f"""
import importlib.util, json, os, tempfile, threading, time
from pathlib import Path

# Load the module by filename to avoid importing pysr/__init__.py
spec = importlib.util.spec_from_file_location('jp_mod', r'''{module_path}''')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Executor = mod.FileBasedProgressExecutor

import ipywidgets as widgets
from IPython.display import display

bar = widgets.IntProgress(value=0, min=0, max=12, description='PySR')
display(bar)

# Minimal widget adapter (uses real ipywidgets -> emits comm messages)
class W:
    def update(self, c, t):
        bar.max = max(int(t), 1)
        bar.value = min(max(int(c), 0), bar.max)
    def set_message(self, msg):
        pass
    def close(self):
        pass

steps = 12
expected = [1,3,7,8,12]
seen = {{c: threading.Event() for c in expected}}

# Hook into updates to release worker per step
orig_update = W.update
def update(self, c, t):
    orig_update(self, c, t)
    c = int(c)
    ev = seen.get(c)
    if ev is not None:
        ev.set()
W.update = update

class MockJulia:
    def __call__(self, *args, progress_file=None, **kwargs):
        p = Path(progress_file)
        def write(cur):
            p.write_text(json.dumps({{'current': int(cur), 'total': steps}}))
        write(1); assert seen[1].wait(2.0)
        write(3); assert seen[3].wait(2.0)
        p.write_text('{'); time.sleep(0.01)
        write(7); assert seen[7].wait(2.0)
        write(8); assert seen[8].wait(2.0)
        write(12); assert seen[12].wait(2.0)
        return 'ok'

ex = Executor()
fd, path = tempfile.mkstemp(suffix='.json'); os.close(fd)
try:
    res = ex.execute(MockJulia(), W(), path, steps, poll_interval_s=0.01)
    print('KERNEL_TEST_RESULT', res)
finally:
    try: os.unlink(path)
    except Exception: pass
"""

        km = KernelManager()
        km.start_kernel()
        try:
            kc = km.client()
            kc.start_channels()

            msg_id = kc.execute(code)
            comm_msgs = 0
            idle = False
            t0 = time.time()

            # Collect messages until idle.
            while not idle and (time.time() - t0) < 30:
                msg = kc.get_iopub_msg(timeout=5)
                msg_type = msg.get("header", {{}}).get("msg_type")
                if msg_type == "comm_msg":
                    comm_msgs += 1
                if msg_type == "status" and msg.get("content", {{}}).get("execution_state") == "idle":
                    idle = True

            self.assertTrue(idle, "kernel never became idle")
            # We expect multiple widget state updates while running.
            self.assertGreaterEqual(comm_msgs, 2, f"expected >=2 comm_msg, got {comm_msgs}")
        finally:
            try:
                kc.stop_channels()
            except Exception:
                pass
            km.shutdown_kernel(now=True)
