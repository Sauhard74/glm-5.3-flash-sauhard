import json

import mlx.core as mx
import numpy as np
import pytest

from evals.kl import eval_sample, iter_ref, kl_terms, write_self_ref


def _exact_kl(lp, lq):
    return (np.exp(lp) * (lp - lq)).sum(-1)


def _logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def test_kl_terms_full_support_is_exact():
    rng = np.random.default_rng(0)
    lp = _logsoftmax(rng.normal(size=(5, 16)) * 2)
    lq = _logsoftmax(rng.normal(size=(5, 16)) * 2)
    idx = np.argsort(-lp, axis=-1).astype(np.int32)
    top_lp = np.take_along_axis(lp, idx, axis=-1)
    kl, agree = kl_terms(mx.array(lq.astype(np.float32)), mx.array(idx), mx.array(top_lp.astype(np.float32)))
    np.testing.assert_allclose(np.array(kl), _exact_kl(lp, lq), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(np.array(agree), lq.argmax(-1) == lp.argmax(-1))


def test_kl_terms_topk_is_lower_bound_and_zero_for_identical():
    rng = np.random.default_rng(1)
    lp = _logsoftmax(rng.normal(size=(8, 200)) * 3)
    lq = _logsoftmax(rng.normal(size=(8, 200)) * 3)
    idx = np.argsort(-lp, axis=-1)[:, :16].astype(np.int32)
    top_lp = np.take_along_axis(lp, idx, axis=-1).astype(np.float32)
    kl, _ = kl_terms(mx.array(lq.astype(np.float32)), mx.array(idx), mx.array(top_lp))
    exact = _exact_kl(lp, lq)
    assert (np.array(kl) <= exact + 1e-4).all()
    assert (np.array(kl) > 0).all()
    same, agree = kl_terms(mx.array(lp.astype(np.float32)), mx.array(idx), mx.array(top_lp))
    assert np.abs(np.array(same)).max() < 1e-5
    assert np.array(agree).all()


def test_reader_matches_calib_format(tmp_path):
    toks = np.arange(10, dtype=np.int32)
    np.savez(
        tmp_path / "00000.npz",
        tokens=toks,
        top_idx=np.zeros((10, 4), np.int32),
        top_logprob=np.zeros((10, 4), np.float32),
        target_logprob=np.full(10, np.nan, np.float32),
    )
    (tmp_path / "index.json").write_text(
        json.dumps({"model": "qwen", "topk": 4, "samples": [{"id": "a", "domain": "code", "file": "00000.npz", "n_tokens": 10}]})
    )
    (s,) = list(iter_ref(tmp_path))
    assert s["id"] == "a" and s["domain"] == "code" and s["top_idx"].shape == (10, 4)


@pytest.mark.parametrize("family", ["qwen", "glm"])
def test_self_reference_kl_is_near_zero(tiny_models, family, tmp_path):
    from mlx_vlm import load

    model, _ = load(str(tiny_models[family]))
    rng = np.random.default_rng(0)
    samples = [
        {"id": "s0", "domain": "code", "tokens": rng.integers(1000, 50000, size=96).tolist()},
        {"id": "s1", "domain": "longctx", "tokens": rng.integers(1000, 50000, size=300).tolist()},
    ]
    ref = write_self_ref(model, samples, tmp_path / "ref")
    for s in iter_ref(ref):
        r = eval_sample(model, s, chunk=64)
        assert r["kl"].shape == (len(s["tokens"]),)
        assert r["kl"].mean() < 1e-4
        assert r["agree"].mean() > 0.9
        ref_nll = -np.nanmean(s["target_logprob"])
        assert abs(r["nll"].mean() - ref_nll) < 1e-2


class _Tok:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv = {v: k for k, v in vocab.items()}

    def get_vocab(self):
        return self.vocab

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


def test_position_roles_glm_markers():
    from evals.kl import position_roles

    tok = _Tok({"<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|observation|>": 4, "x": 5})
    roles = position_roles(np.array([5, 1, 5, 2, 5, 3, 5, 4, 5]), tok)
    assert list(roles) == ["none", "system", "system", "user", "user", "assistant", "assistant", "tool", "tool"]


def test_position_roles_chatml():
    from evals.kl import position_roles

    tok = _Tok({"<|im_start|>": 1, "user": 2, "assistant": 3, "\n": 4, "x": 5})
    roles = position_roles(np.array([1, 2, 4, 5, 1, 3, 4, 5]), tok)
    assert list(roles) == ["none", "user", "user", "user", "user", "assistant", "assistant", "assistant"]
