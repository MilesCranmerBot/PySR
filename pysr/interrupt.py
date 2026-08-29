"""Cooperative SIGINT handling so searches stop gracefully instead of killing the process."""

from __future__ import annotations

import ctypes
import os
import signal
import threading
import warnings
from contextlib import ExitStack, contextmanager

from .julia_import import SymbolicRegression


class _SigactionStorage(ctypes.Union):
    _fields_ = [
        ("alignment", ctypes.c_longdouble),
        ("storage", ctypes.c_ubyte * 1024),
    ]


def _libc_with_sigaction():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.sigaction.restype = ctypes.c_int
    return libc


def _checked_sigaction(libc, action, old_action):
    result = libc.sigaction(signal.SIGINT, action, old_action)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _should_arm_external_stop() -> bool:
    return (
        os.name == "posix"
        and threading.current_thread() is threading.main_thread()
        and os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") == "yes"
    )


@contextmanager
def _external_stop_signal_context(model):
    interrupted = False
    external_stop = None

    with ExitStack() as cleanup:
        if _should_arm_external_stop():
            stop_read_fd, stop_write_fd = os.pipe()
            cleanup.callback(os.close, stop_read_fd)
            cleanup.callback(os.close, stop_write_fd)
            os.set_blocking(stop_read_fd, False)
            os.set_blocking(stop_write_fd, False)
            external_stop = SymbolicRegression.ExternalStop(stop_read_fd, signal.SIGINT)

            libc = _libc_with_sigaction()
            saved_sigaction = _SigactionStorage()
            _checked_sigaction(libc, None, ctypes.byref(saved_sigaction))
            cleanup.callback(
                _checked_sigaction, libc, ctypes.byref(saved_sigaction), None
            )

            saved_python_handler = signal.getsignal(signal.SIGINT)

            def record_interrupt(*_):
                nonlocal interrupted
                interrupted = True

            signal.signal(signal.SIGINT, record_interrupt)
            cleanup.callback(signal.signal, signal.SIGINT, saved_python_handler)
            previous_wakeup_fd = signal.set_wakeup_fd(stop_write_fd)
            cleanup.callback(signal.set_wakeup_fd, previous_wakeup_fd)

        yield external_stop

    model.interrupted_ = interrupted
    if interrupted:
        warnings.warn(
            "The search was interrupted. Returning partial results.",
            RuntimeWarning,
            stacklevel=3,
        )
