import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
import multiprocessing as mp
from pathlib import Path


def _load_jupyter_progress_module() -> object:
    """Load pysr/jupyter_progress.py *without* importing pysr/__init__.py.

    (Importing pysr normally can require juliacall at import time.)
    """
    module_path = Path(__file__).resolve().parents[1] / "jupyter_progress.py"
    spec = importlib.util.spec_from_file_location("_jp_file_based_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _run_executor_child(q, steps: int):
    """Run in a subprocess so the parent test can enforce a hard timeout."""
    try:
        module = _load_jupyter_progress_module()
        Executor = getattr(module, "FileBasedProgressExecutor")

        # We use a per-step handshake so the worker *cannot* finish unless the
        # executor really observes each written value while it's still running.
        expected_currents = [1, 3, 7, 8, steps]
        seen_events: dict[int, threading.Event] = {c: threading.Event() for c in expected_currents}
        seen_order: list[int] = []

        class MockWidget:
            def __init__(self):
                self.updates: list[tuple[int, int]] = []
                self.update_thread_ids: list[int] = []

            def update(self, current: int, total: int) -> None:
                c = int(current)
                t = int(total)
                self.update_thread_ids.append(threading.get_ident())
                self.updates.append((c, t))
                # Record first time we see an expected current.
                ev = seen_events.get(c)
                if ev is not None and not ev.is_set():
                    seen_order.append(c)
                    ev.set()

            def set_message(self, message: str) -> None:
                pass

            def close(self) -> None:
                pass

        worker_thread_id: dict[str, int] = {}

        class MockJuliaFunction:
            """Simulates Julia writing progress updates to a JSON file."""

            def __init__(self, steps: int):
                self.steps = int(steps)

            def _write(self, p: Path, current: int) -> None:
                p.write_text(json.dumps({"current": int(current), "total": self.steps}))

            def __call__(self, *args, progress_file=None, **kwargs):
                worker_thread_id["id"] = threading.get_ident()
                assert progress_file is not None
                p = Path(progress_file)

                # Step 1
                self._write(p, 1)
                if not seen_events[1].wait(timeout=2.0):
                    raise RuntimeError("Never observed current=1 (not live or not reading file)")

                # Step 3
                self._write(p, 3)
                if not seen_events[3].wait(timeout=2.0):
                    raise RuntimeError("Never observed current=3 (not live or not reading file)")

                # Inject a transient invalid JSON write to ensure robustness.
                p.write_text("{")
                time.sleep(0.01)

                # Step 7
                self._write(p, 7)
                if not seen_events[7].wait(timeout=2.0):
                    raise RuntimeError("Never observed current=7")

                # Step 8
                self._write(p, 8)
                if not seen_events[8].wait(timeout=2.0):
                    raise RuntimeError("Never observed current=8")

                # Final
                self._write(p, self.steps)
                if not seen_events[self.steps].wait(timeout=2.0):
                    raise RuntimeError("Never observed final current")

                return "ok"

        w = MockWidget()
        ex = Executor()

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            t0 = time.time()
            res = ex.execute(
                MockJuliaFunction(steps=steps),
                w,
                path,
                steps,
                poll_interval_s=0.01,
            )
            dt = time.time() - t0
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

        q.put(
            {
                "res": res,
                "dt": dt,
                "updates": w.updates,
                "update_thread_ids": w.update_thread_ids,
                "worker_thread_id": worker_thread_id.get("id"),
                "seen_order": seen_order,
                "expected_currents": expected_currents,
            }
        )
    except BaseException as e:
        q.put({"error": repr(e)})


class TestFileBasedProgressExecutor(unittest.TestCase):
    def test_executor_live_updates_from_file_while_running(self):
        # Run in a subprocess to avoid hanging the test suite on regressions.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        steps = 12
        p = ctx.Process(target=_run_executor_child, args=(q, steps))

        p.start()
        p.join(timeout=6.0)
        if p.is_alive():
            p.kill()
            p.join()
            self.fail("FileBasedProgressExecutor.execute() appears to hang (timeout)")

        self.assertEqual(p.exitcode, 0)
        out = q.get(timeout=1.0)
        if "error" in out:
            self.fail(out["error"])

        self.assertEqual(out["res"], "ok")
        self.assertLess(out["dt"], 6.0)

        updates = out["updates"]
        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(updates[-1], (steps, steps))

        # This is the key property: we observed the expected intermediate currents
        # in order (worker blocks until they are observed).
        self.assertEqual(out["seen_order"], out["expected_currents"])

        # Updates must not be coming from the worker thread.
        worker_tid = out["worker_thread_id"]
        self.assertIsNotNone(worker_tid)
        self.assertTrue(all(t != worker_tid for t in out["update_thread_ids"]))
