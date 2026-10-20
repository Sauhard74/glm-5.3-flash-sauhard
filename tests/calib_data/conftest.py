import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE.parents[1], HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calib_data_helpers import tokenizer_dir  # noqa: E402


@pytest.fixture(scope="session")
def renderers():
    from calib_data.render import Renderer

    cache = {}

    def get(model):
        if model not in cache:
            cache[model] = Renderer(model, tokenizer_dir(model))
        return cache[model]

    return get
