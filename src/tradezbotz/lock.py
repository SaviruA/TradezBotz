"""Single-instance lock.

Two ingests running at once is not a hypothetical: it happened, and the cost was
real. Each EDGAR client self-limits to 8 requests/second, comfortably under the
SEC's 10/s ceiling -- but two clients together hit ~16/s, which risks getting the
IP blocked. They also fight over the same SQLite files.

The trap was a backgrounded job reporting "completed" when its wrapper shell
exited rather than when the work finished, so a second run looked safe to start.
A lock removes the need to remember that.

Stale locks are reclaimed automatically: a lock naming a PID that no longer
exists is assumed to be the debris of a killed run.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType


class LockHeld(RuntimeError):
    pass


def _process_alive(pid: int) -> bool:
    """Whether a process with this PID currently exists.

    On POSIX, signal 0 performs the permission and existence checks without
    delivering anything. On Windows there is no such call, so we shell out to
    tasklist; a parse failure is treated as "alive", because wrongly reclaiming a
    live lock is far worse than leaving a stale one for the user to remove.
    """
    if os.name == "nt":
        import subprocess

        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except Exception:  # noqa: BLE001
            return True
        return str(pid) in out

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class SingleInstance:
    """Context manager holding an exclusive lock for `name`."""

    def __init__(self, name: str, directory: str | Path = "state") -> None:
        self.path = Path(directory) / f"{name}.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def acquire(self) -> None:
        if self.path.exists():
            raw = self.path.read_text().strip()
            try:
                pid = int(raw)
            except ValueError:
                pid = -1
            if pid > 0 and _process_alive(pid):
                raise LockHeld(
                    f"Another run is already active (PID {pid}, lock {self.path}).\n"
                    "Running two at once exceeds the SEC's request limit and can "
                    "get the IP blocked. Wait for it, or stop it and retry."
                )
            self.path.unlink(missing_ok=True)  # stale

        # O_EXCL so two processes racing here cannot both believe they won.
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "SingleInstance":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
