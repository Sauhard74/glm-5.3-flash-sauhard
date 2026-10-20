import json

import mlx.core as mx
import numpy as np
import pytest

from build.make import main as make_main
from recover.blockdwq import main as dwq_main
from tests.fixtures.make_tiny import make_glm


def _data(path, vocab, n=96, seed=0):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(path / "calib.jsonl", "w") as f:
        for i in range(n):
            d = ["code", "agentic", "reasoning", "longctx"][i % 4]
            f.write(json.dumps({"id": f"c{i}", "domain": d, "tokens": rng.integers(0, vocab, 96).tolist()}) + "\n")
    return path


def _weights(path):
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        out.update(mx.load(str(f)))
    return out


def _logits(path, tokens):
    from mlx_vlm.utils import load_model

    return load_model(path).language_model(tokens).logits[0].astype(mx.float32)


def _kl(p_logits, q_logits):
    lp = p_logits - mx.logsumexp(p_logits, -1, keepdims=True)
    lq = q_logits - mx.logsumexp(q_logits, -1, keepdims=True)
    return float((mx.exp(lp) * (lp - lq)).sum(-1).mean())


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("dwq")
    src = make_glm(d / "src")
    vocab = json.loads((src / "config.json").read_text())["text_config"]["vocab_size"]
    common = ["--model", "glm", "--src", str(src), "--prune", "none", "--mtp", "drop", "--budget-gb", "1", "--drop-vision"]
    make_main(common + ["--out", str(d / "bf16"), "--pin", ".*=16"])
    # decoder layers at 3 bits; embedding/head left bf16 (DWQ does not train them)
    make_main(common + ["--out", str(d / "q3"), "--pin", r"(embed_tokens|lm_head)$=16",
                        "--pin", r"^(?!.*(\.mlp\.gate|embed_tokens|lm_head)$).*=3:32"])
    return d, src, _data(d / "data", vocab)


def test_blockdwq_trains_only_scales_biases_and_accepts_only_improvements(setup):
    """Mechanics only. On a random tiny model quantization error is unstructured, so end-to-end KL barely moves;
    the real gain has to be measured on the full model (evals.kl)."""
    d, src, data = setup
    out = d / "q3-dwq"
    dwq_main(["--model", "glm", "--src", str(src), "--student", str(d / "q3"), "--data", str(data), "--out", str(out),
              "--tokens", "6144", "--val-tokens", "1536", "--seq-len", "96", "--epochs", "4", "--lr", "3e-3", "--mem-gb", "8"])
    rep = json.loads((out / "dwq_report.json").read_text())
    acc = [r for r in rep["layers"] if r["accepted"]]
    assert acc and all(r["val_after"] < r["val_before"] for r in acc)
    a, b = _weights(d / "q3"), _weights(out)
    assert set(a) == set(b)
    changed = [k for k in a if not mx.array_equal(a[k], b[k]).item()]
    assert changed and all(k.endswith((".scales", ".biases")) for k in changed)
    assert all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape for k in a)
    tokens = mx.array(np.random.default_rng(5).integers(0, 1000, (1, 64)))
    ref = _logits(d / "bf16", tokens)
    assert _kl(ref, _logits(out, tokens)) < 1.05 * _kl(ref, _logits(d / "q3", tokens))
