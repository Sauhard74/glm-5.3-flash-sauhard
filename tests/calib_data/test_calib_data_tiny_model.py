"""Built token streams run through tiny random-initialized glm5_next / qwen4_exp at the real vocab size."""

import json

import mlx.core as mx
import pytest

from calib_data import build
from calib_data_helpers import tokenizer_dir


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_tokens_feed_real_architecture(tmp_path, model, renderers):
    from mlx_vlm.utils import load_model

    from tests.fixtures.make_tiny import make_glm, make_qwen

    vocab = renderers(model).vocab_size
    out = tmp_path / "data"
    build.main(["--model", model, "--out", str(out), "--tokenizer", str(tokenizer_dir(model)), "--smoke"])
    rows = [json.loads(line) for line in (out / "eval.jsonl").read_text().splitlines()]

    src = (make_glm if model == "glm" else make_qwen)(tmp_path / "tiny", vocab_size=vocab)
    lm = load_model(src).language_model
    for r in rows:
        toks = mx.array([r["tokens"][:256]])
        logits = lm(toks, cache=lm.make_cache()).logits
        assert logits.shape == (1, toks.shape[1], vocab)
        assert not mx.isnan(logits).any().item()
