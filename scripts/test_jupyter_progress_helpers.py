"""Standalone tests for jupyter_progress helpers (no juliacall import required)."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "pysr" / "jupyter_progress.py"
    spec = importlib.util.spec_from_file_location("_jp_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_extracts_progress() -> None:
    module = _load_module()
    updates = []
    parser = module._ProgressLineParser(lambda current, total: updates.append((current, total)))
    parser.parse_line("Progress: 2 / 5 total iterations")
    assert updates == [(2, 5)], updates


def test_capture_stream_handles_split_lines() -> None:
    module = _load_module()
    updates = []
    parser = module._ProgressLineParser(lambda current, total: updates.append((current, total)))
    capture = module._ProgressCaptureStream(io.StringIO(), parser)
    capture.write("Progress: 1 / ")
    capture.write("3 total iterations\n")
    capture.flush()
    assert updates == [(1, 3)], updates


def test_parser_extracts_evolving_percent() -> None:
    module = _load_module()
    updates = []
    parser = module._ProgressLineParser(lambda current, total: updates.append((current, total)))
    parser.parse_line("Evolving for 40 iterations... 11%|██        | ETA: 0:00:05")
    assert updates == [(4, 40)], updates


def test_capture_stream_handles_carriage_return_updates() -> None:
    module = _load_module()
    updates = []
    parser = module._ProgressLineParser(lambda current, total: updates.append((current, total)))
    capture = module._ProgressCaptureStream(io.StringIO(), parser)
    capture.write("Evolving for 100 iterations... 1%|\r")
    capture.write("Evolving for 100 iterations... 2%|\r")
    capture.write("Evolving for 100 iterations... 3%|\r")
    capture.flush()
    assert updates == [(1, 100), (2, 100), (3, 100)], updates


def test_capture_stream_handles_no_delimiter_updates() -> None:
    module = _load_module()
    updates = []
    parser = module._ProgressLineParser(lambda current, total: updates.append((current, total)))
    capture = module._ProgressCaptureStream(io.StringIO(), parser)
    capture.write("Progress: 4 / 100 total iterations")
    capture.write(" ... Progress: 5 / 100 total iterations")
    capture.write(" ... Progress: 6 / 100 total iterations")
    assert updates[-1] == (6, 100), updates


def run() -> None:
    test_parser_extracts_progress()
    test_capture_stream_handles_split_lines()
    test_parser_extracts_evolving_percent()
    test_capture_stream_handles_carriage_return_updates()
    test_capture_stream_handles_no_delimiter_updates()
    print("helpers-tests=ok")


if __name__ == "__main__":
    run()
