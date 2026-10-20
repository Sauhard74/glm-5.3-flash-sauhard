import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.make_tiny import make_glm, make_qwen  # noqa: E402

DOMAINS = ["code", "agentic", "reasoning", "longctx"]


@pytest.fixture(scope="session")
def tiny_glm(tmp_path_factory):
    return make_glm(tmp_path_factory.mktemp("tiny") / "glm")


@pytest.fixture(scope="session")
def tiny_qwen(tmp_path_factory):
    return make_qwen(tmp_path_factory.mktemp("tiny") / "qwen")


@pytest.fixture(scope="session")
def tiny(tiny_glm, tiny_qwen):
    return {"glm": tiny_glm, "qwen": tiny_qwen}


def write_data(path: Path, vocab=1000, seed=0, n_calib=8, n_eval=3, long_len=420):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(path / "calib.jsonl", "w") as f:
        for i in range(n_calib):
            d = DOMAINS[i % 4]
            n = long_len if d == "longctx" else int(rng.integers(40, 160))
            f.write(json.dumps({"id": f"c{i}", "domain": d, "tokens": rng.integers(0, vocab, n).tolist()}) + "\n")
    with open(path / "eval.jsonl", "w") as f:
        for i in range(n_eval):
            n = long_len if i == 0 else int(rng.integers(30, 120))
            f.write(json.dumps({"id": f"e{i}", "domain": DOMAINS[i % 4], "tokens": rng.integers(0, vocab, n).tolist()}) + "\n")
    return path


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return write_data(tmp_path_factory.mktemp("data"))
