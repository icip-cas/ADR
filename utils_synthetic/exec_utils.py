"""Execution helpers shared across pipeline scripts.

Currently provides a signal-based wall-clock timeout used to guard untrusted
generated code (test-case generators) during validation.
"""
import signal
import contextlib


class TimeoutException(Exception):
    """Raised when a guarded block exceeds its allotted wall-clock time."""
    pass


@contextlib.contextmanager
def timeout(seconds, error_message="Execution timed out"):
    """Raise TimeoutException after `seconds` (SIGALRM-based).

    Relies on signal.SIGALRM, so it only works on the main thread of a Unix
    process. Pipeline step 5 runs it inside ProcessPoolExecutor workers, where
    each worker's main thread is valid.
    """
    def _handle_timeout(signum, frame):
        raise TimeoutException(error_message)

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
