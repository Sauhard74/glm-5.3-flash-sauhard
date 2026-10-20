import os
import time
from pathlib import Path

import pytest

from evals import code as code_suite
from evals.data import humaneval_plus, mbpp_plus
from evals.sandbox import run_python, sandbox_available


def test_pass_and_fail():
    assert run_python("print('hi')").status == "pass"
    r = run_python("assert 1 == 2")
    assert r.status == "fail" and "AssertionError" in r.output


def test_timeout_kills_process_group():
    t = time.time()
    r = run_python("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'while True: pass'])\nwhile True: pass", timeout=2)
    assert r.status == "timeout"
    assert time.time() - t < 10


def test_network_blocked():
    r = run_python("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)")
    assert r.status == "fail"


@pytest.mark.skipif(not sandbox_available(), reason="sandbox-exec not available")
def test_writes_outside_workdir_blocked(tmp_path):
    target = Path.home() / f".evals-sandbox-probe-{os.getpid()}"
    r = run_python(f"open({str(target)!r}, 'w').write('x')")
    assert r.status == "fail"
    assert not target.exists()
    assert run_python("open('ok.txt', 'w').write('x')").status == "pass"


def test_canonical_solutions_pass_and_wrong_fail():
    items = humaneval_plus(smoke=True) + mbpp_plus(smoke=True)
    good = [code_suite.oracle_completion(it) for it in items]
    res = code_suite.score(items, good, timeout=20, workers=4)
    assert all(r["plus"] == "pass" for r in res)
    assert all(r.get("base", "pass") == "pass" for r in res)
    bad = ["```python\ndef add(a, b):\n    return a - b\n```"] + ["no code at all"] * (len(items) - 1)
    res = code_suite.score(items, bad, timeout=20, workers=4)
    assert all(r["plus"] == "fail" for r in res)


def test_humaneval_body_only_completion_gets_prompt_prepended():
    item = humaneval_plus(smoke=True)[0]
    progs = code_suite.assemble(item, "    return a + b\n")
    assert run_python(progs["plus"]).status == "pass"
