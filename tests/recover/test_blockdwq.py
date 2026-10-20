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


def _args(d, src, data, out, *extra):
    return ["--model", "glm", "--src", str(src), "--student", str(d / "q3"), "--data", str(data), "--out", str(out),
            "--tokens", "3072", "--val-tokens", "768", "--seq-len", "96", "--epochs", "2", "--lr", "3e-3",
            "--mem-gb", "8", *extra]


def test_split_frozen_attention_matches_full_forward(setup):
    from recover.blockdwq import BlockDWQ, frozen_branches, parse_args, split_frozen_attention

    d, src, data = setup
    b = BlockDWQ(parse_args(_args(d, src, data, d / "unused")))
    h = b.student.embed(b.train[0])[None]
    split_layers = 0
    for i in range(b.teacher.n_layers):
        layer = b.student.layer(i)
        full = b.student.arch.forward(layer, h, None, None)
        split = split_frozen_attention(layer)
        assert (split is None) == (not frozen_branches(layer))
        if split is not None:
            attn, ffn = split
            np.testing.assert_array_equal(np.array(ffn(attn(h)).astype(mx.float32)), np.array(full.astype(mx.float32)))
            split_layers += 1
        h = full
    assert split_layers >= 2  # the tiny GLM has linear- and sparse-attention MoE layers


def test_resume_after_kill_is_identical(setup, monkeypatch):
    import recover.blockdwq as bd

    d, src, data = setup
    bd.main(_args(d, src, data, d / "full"))
    orig = bd.BlockDWQ.train_layer

    def dies_at_2(self, i, *a):
        if i == 2:
            raise KeyboardInterrupt("killed")
        return orig(self, i, *a)

    monkeypatch.setattr(bd.BlockDWQ, "train_layer", dies_at_2)
    with pytest.raises(KeyboardInterrupt):
        bd.main(_args(d, src, data, d / "resumed"))
    assert json.loads((d / "resumed.ckpt" / "state.json").read_text())["next"] == 2
    monkeypatch.setattr(bd.BlockDWQ, "train_layer", orig)
    with pytest.raises(SystemExit):  # different arguments must not silently reuse the checkpoint
        bd.main(_args(d, src, data, d / "resumed", "--resume", "--lr", "1e-3"))
    bd.main(_args(d, src, data, d / "resumed", "--resume"))
    assert not (d / "resumed.ckpt").exists()
    a, b = _weights(d / "full"), _weights(d / "resumed")
    assert set(a) == set(b) and all(mx.array_equal(a[k], b[k]).item() for k in a)
    ra = json.loads((d / "full" / "dwq_report.json").read_text())["layers"]
    rb = json.loads((d / "resumed" / "dwq_report.json").read_text())["layers"]
    assert [(r["layer"], r["accepted"], r.get("val_after")) for r in ra] == \
           [(r["layer"], r["accepted"], r.get("val_after")) for r in rb]


def test_layers_do_not_leak_memory(setup):
    """MLX keeps a custom function's inputs alive; without ExpertGrads.release() every trained MoE layer stayed
    resident (~1.4 GB per GLM layer)."""
    import gc

    from recover.blockdwq import BlockDWQ, _detach_attention, _no_index_grad, parse_args
    from recover.qgrad import ExpertGrads

    d, src, data = setup
    b = BlockDWQ(parse_args(_args(d, src, data, d / "unused")))
    _no_index_grad()
    _detach_attention()
    b.expert_grads = ExpertGrads()
    hs = [b.student.embed(s)[None] for s in b.train + b.val]
    hs = b.fwd(b.student.arch, b.student.layer(0), hs)
    n = len(b.train)
    used = []
    for _ in range(6):
        layer = b.student.layer(1)
        b.train_layer(1, layer, hs[:n], hs[:n], hs[n:], hs[n:])
        b.fwd(b.student.arch, layer, hs)
        del layer
        b.expert_grads.release()
        b.updates.clear()
        gc.collect()
        mx.clear_cache()
        used.append(mx.get_active_memory())
    assert used[-1] <= used[1] * 1.01
