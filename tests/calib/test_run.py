import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from calib import sens as S
from calib.arch import make_arch
from calib.run import chunk_bounds, main


def _run(arch, src, data, out, *extra):
    main(["--model", arch, "--src", str(src), "--data", str(data), "--out", str(out), *extra])
    return Path(out)


def _full_lm(arch_name, src):
    arch = make_arch(arch_name, src, mx.float32)
    full = arch.module.Model(arch.model_config)
    w = full.sanitize(arch.snap.load_dequantized(arch.snap.keys(), dtype=mx.float32))
    w = {
        k[len("language_model."):]: v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v
        for k, v in w.items()
        if k.startswith("language_model.")
    }
    full.language_model.load_weights(list(w.items()), strict=True)
    full.eval()
    return full.language_model


def _ref_vs_model(out: Path, lm, chunk=None):
    index = json.loads((out / "ref/index.json").read_text())
    worst = 0.0
    for s in index["samples"]:
        z = np.load(out / "ref" / s["file"])
        toks = z["tokens"]
        cache = lm.make_cache()
        bounds = chunk_bounds(len(toks), chunk) if chunk else [(0, len(toks))]
        logits = mx.concatenate(
            [lm(mx.array(toks[a:b])[None], cache=cache).logits[0].astype(mx.float32) for a, b in bounds]
        )
        lp = np.array(logits - mx.logsumexp(logits, -1, keepdims=True))
        worst = max(worst, np.abs(np.take_along_axis(lp, z["top_idx"], -1) - z["top_logprob"]).max())
        assert np.allclose(np.take_along_axis(lp[:-1], toks[1:, None], -1)[:, 0],
                           z["target_logprob"][:-1], atol=1e-5)
        assert np.all(np.diff(z["top_logprob"], axis=-1) <= 0)
    return worst


@pytest.mark.parametrize("arch", ["glm", "qwen"])
def test_streaming_matches_full_model(tiny, data_dir, tmp_path, arch):
    lm = _full_lm(arch, tiny[arch])
    f32 = ("--dtype", "float32", "--store-dtype", "float32")
    out = _run(arch, tiny[arch], data_dir, tmp_path / "one", "--chunk", "4096", *f32)
    assert _ref_vs_model(out, lm) < 1e-4
    out = _run(arch, tiny[arch], data_dir, tmp_path / "chunked", "--chunk", "64", *f32)
    assert _ref_vs_model(out, lm, chunk=64) < 1e-4


@pytest.mark.parametrize("arch", ["glm", "qwen"])
def test_outputs_and_resume(tiny, data_dir, tmp_path, arch):
    src = tiny[arch]
    full = _run(arch, src, data_dir, tmp_path / "full", "--chunk", "128")
    a = make_arch(arch, src)
    n = a.n_layers
    meta = json.loads((full / "meta.json").read_text())
    assert meta["num_layers"] == n and meta["domains"] == ["code", "agentic", "reasoning", "longctx"]
    assert json.loads((full / "state.json").read_text())["done"]
    assert not list((full / "work").iterdir())

    saw_moe = False
    for i in range(n):
        m = json.loads((full / f"layer_{i:03d}.json").read_text())
        z = np.load(full / f"layer_{i:03d}.npz")
        assert m["layer"] == i and m["prefix"] == f"language_model.model.layers.{i}."
        layer = a.load_layer(i)
        modules = dict(layer.named_modules())
        for name, t in m["tensors"].items():
            if t["kind"] != "embedding":
                assert list(modules[name].weight.shape) == t["shape"]
            assert set(t["sens"]) == {S.grid_key(b, g) for b in S.BITS for g in S.GROUPS}
            col = [t["sens"][S.grid_key(b, 32)] for b in S.BITS]
            if None not in col:
                assert all(x > y for x, y in zip(col, col[1:])), (name, col)
            if t["kind"] in ("linear", "switch") or f"act_absmean.{name}" in z.files:
                assert z[f"act_absmean.{name}"].shape[-1] == t["shape"][-1]
        if m["moe"]:
            saw_moe = True
            E = m["moe"]["n_experts"]
            assert np.isclose(sum(m["moe"]["expert_freq"]), 1.0)
            assert z["sens.mlp.switch_mlp.down_proj"].shape == (E, len(S.BITS), len(S.GROUPS))
            assert z["act_absmean.mlp.switch_mlp.gate_proj"].shape == (E, a.hidden_size)
            for d in meta["domains"]:
                assert np.isclose(z[f"expert_freq.{d}"].sum(), 1.0)
                assert z[f"expert_reap.{d}"].shape == (E,)
            assert np.all(z["expert_reap"] >= 0) and z["expert_reap"].max() > 0
    assert saw_moe
    for name in ("embed", "head"):
        assert json.loads((full / f"{name}.json").read_text())["tensors"]
    index = json.loads((full / "ref/index.json").read_text())
    assert [s["id"] for s in index["samples"]] == ["e0", "e1", "e2"]

    part = tmp_path / "part"
    _run(arch, src, data_dir, part, "--chunk", "128", "--layers", "1")
    assert json.loads((part / "state.json").read_text()) == {"next_layer": 1}
    assert (part / "work/h_001").is_dir() and not (part / "layer_001.json").exists()
    _run(arch, src, data_dir, part, "--chunk", "128")
    for i in range(n):
        za, zb = np.load(full / f"layer_{i:03d}.npz"), np.load(part / f"layer_{i:03d}.npz")
        assert za.files == zb.files
        for k in za.files:  # GPU scatter-add order makes act stats differ at ~1e-7
            assert np.allclose(za[k], zb[k], rtol=1e-5, atol=0, equal_nan=True), (i, k)
    for s in index["samples"]:
        za, zb = np.load(full / "ref" / s["file"]), np.load(part / "ref" / s["file"])
        assert np.array_equal(za["top_logprob"], zb["top_logprob"])


def test_meta_mismatch_requires_restart(tiny, data_dir, tmp_path):
    out = _run("qwen", tiny["qwen"], data_dir, tmp_path / "o", "--layers", "1")
    other = tmp_path / "d2"
    shutil.copytree(data_dir, other)
    with open(other / "calib.jsonl", "a") as f:
        f.write(json.dumps({"id": "x", "domain": "code", "tokens": [1, 2, 3, 4]}) + "\n")
    with pytest.raises(SystemExit):
        _run("qwen", tiny["qwen"], other, out)
    _run("qwen", tiny["qwen"], other, out, "--restart", "--layers", "1")
    assert json.loads((out / "meta.json").read_text())["n_samples"]["calib"] == 9


def test_wrong_model_flag_rejected(tiny):
    with pytest.raises(ValueError):
        make_arch("glm", tiny["qwen"])
