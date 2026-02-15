import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "jupyter_progress.py"
    spec = importlib.util.spec_from_file_location("_jp_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestJupyterProgressHelpers(unittest.TestCase):
    def test_progress_line_parser_extracts_progress(self):
        module = _load_module()
        updates = []
        parser = module._ProgressLineParser(
            lambda current, total: updates.append((current, total))
        )
        parser.parse_line("Progress: 7 / 13 total iterations (53.8%)")
        self.assertEqual(updates, [(7, 13)])

    def test_progress_capture_stream_handles_partial_lines(self):
        module = _load_module()
        updates = []
        parser = module._ProgressLineParser(
            lambda current, total: updates.append((current, total))
        )
        target = io.StringIO()
        stream = module._ProgressCaptureStream(target, parser)
        stream.write("prefix\nProgress: 4 / ")
        stream.write("10 total iterations\n")
        stream.flush()
        self.assertIn("prefix", target.getvalue())
        self.assertEqual(updates, [(4, 10)])

    def test_progress_parser_extracts_evolving_percent_lines(self):
        module = _load_module()
        updates = []
        parser = module._ProgressLineParser(
            lambda current, total: updates.append((current, total))
        )
        parser.parse_line("Evolving for 40 iterations... 11%|██        | ETA: 0:00:05")
        self.assertEqual(updates, [(4, 40)])

    def test_progress_capture_stream_handles_carriage_return_updates(self):
        module = _load_module()
        updates = []
        parser = module._ProgressLineParser(
            lambda current, total: updates.append((current, total))
        )
        target = io.StringIO()
        stream = module._ProgressCaptureStream(target, parser)
        stream.write("Evolving for 100 iterations... 1%|\r")
        stream.write("Evolving for 100 iterations... 2%|\r")
        stream.write("Evolving for 100 iterations... 3%|\r")
        stream.flush()
        self.assertEqual(updates, [(1, 100), (2, 100), (3, 100)])

    def test_progress_capture_stream_handles_no_delimiter_updates(self):
        module = _load_module()
        updates = []
        parser = module._ProgressLineParser(
            lambda current, total: updates.append((current, total))
        )
        target = io.StringIO()
        stream = module._ProgressCaptureStream(target, parser)
        stream.write("Progress: 4 / 100 total iterations")
        stream.write(" ... Progress: 5 / 100 total iterations")
        stream.write(" ... Progress: 6 / 100 total iterations")
        self.assertEqual(updates[-1], (6, 100))

    def test_should_use_jupyter_progress_gating(self):
        module = _load_module()
        self.assertFalse(
            module.should_use_jupyter_progress(
                progress=False, verbosity=1, is_single_output=True
            )
        )
        self.assertFalse(
            module.should_use_jupyter_progress(
                progress=True, verbosity=0, is_single_output=True
            )
        )
        self.assertFalse(
            module.should_use_jupyter_progress(
                progress=True, verbosity=1, is_single_output=False
            )
        )

    def test_should_use_jupyter_progress_respects_disable_flag(self):
        module = _load_module()
        prev = os.environ.get("PYSR_DISABLE_JUPYTER_PROGRESS")
        os.environ["PYSR_DISABLE_JUPYTER_PROGRESS"] = "1"
        try:
            self.assertFalse(
                module.should_use_jupyter_progress(
                    progress=True, verbosity=1, is_single_output=True
                )
            )
        finally:
            if prev is None:
                os.environ.pop("PYSR_DISABLE_JUPYTER_PROGRESS", None)
            else:
                os.environ["PYSR_DISABLE_JUPYTER_PROGRESS"] = prev

    def test_capture_patches_existing_stream_write_method(self):
        module = _load_module()
        updates = []
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        context = module.JupyterProgressContext(total_iterations=10)
        context._parser.on_progress = lambda current, total: updates.append((current, total))

        try:
            with context.capture():
                sys.stdout.write("Progress: 4 / 10 total iterations\n")
        finally:
            sys.stdout = old_stdout

        self.assertEqual(updates[-1], (4, 10))
        self.assertIs(sys.stdout, old_stdout)




def runtests(just_tests=False):
    tests = [TestJupyterProgressHelpers]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
