"""Run untrusted generated code in a subprocess: no network, writes confined to
a temp dir (macOS sandbox-exec when available), CPU/file-size rlimits, wall
timeout that kills the whole process group."""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

_PRELUDE = """\
import resource as _r, socket as _s
for _lim, _val in ((_r.RLIMIT_CPU, {cpu}), (_r.RLIMIT_FSIZE, 64 << 20), (_r.RLIMIT_CORE, 0)):
    try:
        _r.setrlimit(_lim, (_val, _val))
    except (ValueError, OSError):
        pass
def _no_net(*a, **k):
    raise OSError("network disabled in sandbox")
_s.socket.connect = _no_net
_s.socket.connect_ex = _no_net
_s.create_connection = _no_net
_s.getaddrinfo = _no_net
del _r, _s, _lim, _val
"""


@dataclass
class RunResult:
    status: str  # pass | fail | timeout | error
    returncode: Optional[int]
    output: str


def _profile(workdir: str) -> str:
    return (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f'(allow file-write* (subpath "{workdir}") (literal "/dev/null") (subpath "/dev/fd"))'
    )


def sandbox_available() -> bool:
    return sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC)


def run_python(code: str, timeout: float = 20.0, python: Optional[str] = None) -> RunResult:
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="evals-sbx-"))
    try:
        script = os.path.join(workdir, "prog.py")
        with open(script, "w") as f:
            f.write(_PRELUDE.format(cpu=max(1, int(timeout) + 1)))
            f.write(code)
        cmd: List[str] = [python or sys.executable, "-I", "-B", script]
        if sandbox_available():
            cmd = [SANDBOX_EXEC, "-p", _profile(workdir)] + cmd
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            out, _ = proc.communicate()
            return RunResult("timeout", None, _tail(out))
        status = "pass" if proc.returncode == 0 else "fail"
        return RunResult(status, proc.returncode, _tail(out))
    except OSError as exc:
        return RunResult("error", None, str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _tail(out: Optional[bytes], n: int = 2000) -> str:
    if not out:
        return ""
    return out.decode("utf-8", "replace")[-n:]


def run_many(programs: Sequence[str], timeout: float = 20.0, workers: int = 4) -> List[RunResult]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda c: run_python(c, timeout), programs))
