"""Run model-written Python in a child process for the ``use_code`` tool.

This is NOT a security sandbox. The code runs as the current user with full
access to the file system and network. The only limits are a wall-clock
timeout, a cap on captured output, isolated mode (``python -I``: no user
site-packages, no ``PYTHON*`` environment variables, script directory not on
``sys.path``) and a fresh temporary working directory. Only enable
``use_code`` on a machine or container where running untrusted code is
acceptable.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import IO

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_OUTPUT_CHARS = 10_000


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False
    truncated: bool = False

    def as_observation(self) -> str:
        """Text for the ``Execution Output:`` part of the observation."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append("stderr:\n" + self.stderr.rstrip())
        if self.timed_out:
            parts.append("Execution stopped: time limit reached.")
        if self.truncated:
            parts.append("Output truncated: size limit reached.")
        if not parts:
            parts.append(f"(no output, exit code {self.returncode})")
        return "\n".join(parts)


def _drain(stream: IO[bytes], buf: bytearray, limit: int, over: threading.Event) -> None:
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        room = limit - len(buf)
        if room > 0:
            buf.extend(chunk[:room])
        if len(chunk) > room:
            over.set()


def run_python(
    code: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> SandboxResult:
    """Run ``code`` with ``sys.executable -I -X utf8 -c`` and capture its output.

    The child is killed when ``timeout`` seconds pass or when stdout or stderr
    goes past ``max_output_chars`` bytes. Output is decoded as UTF-8.
    """
    limit = max(1, int(max_output_chars))
    out_buf, err_buf = bytearray(), bytearray()
    over = threading.Event()
    with tempfile.TemporaryDirectory(prefix="cogtools_code_") as workdir:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-X", "utf8", "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
        )
        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            threading.Thread(target=_drain, args=(proc.stdout, out_buf, limit, over), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err_buf, limit, over), daemon=True),
        ]
        for t in readers:
            t.start()

        timed_out = False
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if over.is_set():
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for t in readers:
            t.join(timeout=5)

    return SandboxResult(
        stdout=out_buf.decode("utf-8", errors="replace"),
        stderr=err_buf.decode("utf-8", errors="replace"),
        returncode=None if timed_out else proc.returncode,
        timed_out=timed_out,
        truncated=over.is_set(),
    )


__all__ = ["DEFAULT_MAX_OUTPUT_CHARS", "DEFAULT_TIMEOUT_S", "SandboxResult", "run_python"]
