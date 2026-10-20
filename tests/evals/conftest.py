import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tiny_models import build_tiny  # noqa: E402


@pytest.fixture(scope="session")
def tiny_qwen_model(tmp_path_factory):
    return build_tiny("qwen", tmp_path_factory.mktemp("evals-tiny") / "qwen")


@pytest.fixture(scope="session")
def tiny_glm_model(tmp_path_factory):
    return build_tiny("glm", tmp_path_factory.mktemp("evals-tiny") / "glm")


@pytest.fixture(scope="session")
def tiny_models(tiny_qwen_model, tiny_glm_model):
    return {"qwen": tiny_qwen_model, "glm": tiny_glm_model}
