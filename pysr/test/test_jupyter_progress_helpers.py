import importlib.util
import io
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
