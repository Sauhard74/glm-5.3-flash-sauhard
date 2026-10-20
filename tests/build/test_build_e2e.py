import json
import re

import mlx.core as mx
import numpy as np
import pytest

from build.make import main as make_main
from build.pipeline import GB

FAMILIES = ["qwen", "glm"]
EXPERT_KEY = re.compile(r"(?:^|\.)(?:layers\.(\d+)\.|mtp_block\.)mlp\.(switch_mlp\.\w+\.weight|gate\.weight|gate\.e_score_correction_bias)$")


def run(src, out, *extra):
    return make_main(["--src", str(src), "--out", str(out), *map(str, extra)])


def load_dir(path):
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    return weights


def assert_matches_oracle(built, expected, keep=None, layer_of=None, skip_prefix=None):
    keys = {k for k in expected if not (skip_prefix and k.startswith(skip_prefix))}
    assert set(built) == keys
    for k in keys:
        exp = expected[k]
        if keep is not None and EXPERT_KEY.search(k):
            exp = exp[mx.array(keep[layer_of(k)])]
        got = built[k]
        assert got.shape == exp.shape, k
        assert mx.array_equal(got, exp.astype(got.dtype)).item(), k


def main_layer(k):
    return re.search(r"layers\.(\d+)\.", k).group(1)


def generate_tokens(path, n=6):
    from mlx_vlm import generate, load

    model, processor = load(str(path))
    out = generate(model, processor, "Write a haiku about memory.", max_tokens=n, temperature=0.0, verbose=False)
    return model, out


@pytest.mark.parametrize("family", FAMILIES)
def test_unpruned_bf16_matches_oracle(tiny, tmp_path, family):
    """Streaming read + FP8 dequant + chunked sanitize reproduces mlx-vlm's own conversion bit-exactly."""
    t = tiny(family)
    out = tmp_path / "m"
    run(t["src"], out, "--model", family, "--prune", "none", "--pin", ".*=16", "--budget-gb", 1)
    assert_matches_oracle(load_dir(out), t["expected"])
    assert_matches_oracle(load_dir(out / "mtp"), t["expected_mtp"])
    cfg = json.loads((out / "config.json").read_text())
    assert "quantization_config" in cfg and cfg["quantization_config"].get("quant_method") is None


@pytest.mark.parametrize("family", FAMILIES)
def test_uniform_prune_slices_router_and_experts_consistently(tiny, tmp_path, family):
    t = tiny(family)
    out = tmp_path / "m"
    s = run(t["src"], out, "--model", family, "--calib", t["calib"], "--prune", 0.5, "--pin", ".*=16", "--budget-gb", 1)
    keep = {n: v["kept_ids"] for n, v in s["prune"]["layers"].items()}
    assert {len(v) for v in keep.values()} == {8}
    assert_matches_oracle(load_dir(out), t["expected"], keep, main_layer)
    assert_matches_oracle(load_dir(out / "mtp"), t["expected_mtp"], keep, lambda _k: "mtp")
    cfg = json.loads((out / "config.json").read_text())
    attr = "num_experts" if family == "qwen" else "n_routed_experts"
    assert cfg["text_config"][attr] == 8 and "model_file" not in cfg
    mcfg = json.loads((out / "mtp" / "config.json").read_text())
    assert mcfg["text_config"][attr] == 8
    stats = json.loads((t["calib"] / "layer_003.json").read_text())
    reap = np.array(stats["expert_reap"])
    assert sorted(np.argsort(-reap)[:8].tolist()) == keep["3"]


@pytest.mark.parametrize("family", FAMILIES)
def test_mixed_quant_budget_load_generate(tiny, tmp_path, family):
    from mlx_vlm.speculative.drafters import load_drafter

    t = tiny(family)
    out = tmp_path / "m"
    budget = 0.030 if family == "qwen" else 0.022
    s = run(t["src"], out, "--model", family, "--calib", t["calib_npz"], "--prune", 0.5, "--budget-gb", budget)
    assert s["predicted_total_bytes"] <= budget * GB
    assert s["written_bytes"] == s["predicted_bytes"]
    on_disk = sum(json.loads((d / "model.safetensors.index.json").read_text())["metadata"]["total_size"] for d in (out, out / "mtp"))
    assert on_disk == s["predicted_total_bytes"]
    labels = set(s["bits_histogram_units"])
    assert len(labels - {"bf16"}) >= 2, labels
    for a in s["allocation"]:
        if a["kind"] == "router":
            assert a["bits"] == 16
        if a["kind"] == "lm_head":
            assert a["bits"] >= 6
    cfg = json.loads((out / "config.json").read_text())
    q = cfg["quantization"]
    weights = load_dir(out)
    for path, spec in q.items():
        if isinstance(spec, dict):
            assert f"{path}.scales" in weights, path
            w = weights[f"{path}.weight"]
            assert w.shape[-1] * 32 // spec["bits"] == weights[f"{path}.scales"].shape[-1] * spec["group_size"]
        elif spec is False:
            assert f"{path}.scales" not in weights
    model, out_gen = generate_tokens(out)
    assert out_gen.generation_tokens == 6
    draft, kind = load_drafter(str(out / "mtp"))
    assert kind == "mtp"


@pytest.mark.parametrize("family", FAMILIES)
def test_global_nonuniform_prune_loads_with_shim(tiny, tmp_path, family):
    t = tiny(family)
    out = tmp_path / "m"
    s = run(
        t["src"], out, "--model", family, "--calib", t["calib"], "--prune", 0.5, "--prune-mode", "global",
        "--min-keep", 0.25, "--budget-gb", 0.05, "--drop-vision",
    )
    counts = {n: v for n, v in s["prune"]["experts_per_layer"].items() if n != "mtp"}
    assert sum(counts.values()) == 8 * len(counts)
    assert min(counts.values()) >= 4
    cfg = json.loads((out / "config.json").read_text())
    assert len(set(counts.values())) > 1
    assert cfg["model_file"] == "pruned_moe.py" and (out / "pruned_moe.py").exists()
    assert [c for c in cfg["expert_counts"] if c] == [counts[str(i)] for i in sorted(map(int, counts))]
    weights = load_dir(out)
    assert not any(k.startswith("vision_tower") for k in weights)
    model, out_gen = generate_tokens(out)
    assert model.vision_tower is None
    assert out_gen.generation_tokens == 6
    layers = model.language_model.model.layers
    for name, c in counts.items():
        assert layers[int(name)].mlp.switch_mlp.gate_proj.weight.shape[0] == c


@pytest.mark.parametrize("family", FAMILIES)
def test_auto_prune_and_dry_run(tiny, tmp_path, family):
    t = tiny(family)
    out = tmp_path / "m"
    budget = 0.024 if family == "qwen" else 0.016
    s = run(t["src"], out, "--model", family, "--calib", t["calib"], "--prune", "auto", "--auto-step", 4,
            "--budget-gb", budget, "--dry-run")
    assert s["dry_run"] and not out.exists()
    trace = s["prune"]["auto_trace"]
    feasible = [r for r in trace if r["feasible"]]
    assert feasible
    best = min(feasible, key=lambda r: r["objective"])
    assert set(s["prune"]["experts_per_layer"].values()) - {s["mtp"]["experts"]} <= {best["keep"]}
    assert s["predicted_total_bytes"] <= budget * GB


def test_no_calib_weight_saliency_and_mtp_modes(tiny, tmp_path):
    t = tiny("glm")
    s = run(t["src"], tmp_path / "a", "--model", "glm", "--prune", 8, "--mtp", "intact", "--budget-gb", 0.05)
    assert s["mtp"]["experts"] == 16
    assert all(v["score_source"] == "weights" for v in s["prune"]["layers"].values())
    s = run(t["src"], tmp_path / "b", "--model", "glm", "--prune", 8, "--mtp", "drop", "--budget-gb", 0.05)
    assert s["mtp"] is None and not (tmp_path / "b" / "mtp").exists()


def test_infeasible_budget_reports_minimum(tiny, tmp_path):
    t = tiny("qwen")
    with pytest.raises(ValueError, match="smallest possible build"):
        run(t["src"], tmp_path / "x", "--model", "qwen", "--prune", "none", "--budget-gb", 0.001, "--dry-run")


@pytest.mark.parametrize("family", FAMILIES)
def test_mixed_build_equals_inmemory_quantization(tiny, tmp_path, family):
    """The written mixed-precision model == the bf16 build quantized in memory with the same per-tensor config."""
    import mlx.nn as nn
    from mlx_vlm.utils import load_model

    t = tiny(family)
    ref_dir, mixed_dir, q8_dir = tmp_path / "ref", tmp_path / "mixed", tmp_path / "q8"
    common = ["--model", family, "--calib", t["calib"], "--prune", 0.5, "--mtp", "drop"]
    run(t["src"], ref_dir, *common, "--budget-gb", 1, "--pin", ".*=16")
    budget = 0.024 if family == "qwen" else 0.016
    s = run(t["src"], mixed_dir, *common, "--budget-gb", budget)
    run(t["src"], q8_dir, *common, "--budget-gb", 1, "--pin", r"^(?!.*\.mlp\.gate$).*=8:32", "--no-bf16")
    assert len({(a["bits"], a["group_size"]) for a in s["allocation"]}) >= 3
    tokens = mx.array([[9707, 11, 1246, 525, 498, 30, 358, 1079, 4960, 13, 576, 4226]])

    def logits(model):
        return model.language_model(tokens).logits[0].astype(mx.float32)

    ref = logits(load_model(ref_dir))
    mixed = logits(load_model(mixed_dir))
    qcfg = json.loads((mixed_dir / "config.json").read_text())["quantization"]
    inmem = load_model(ref_dir)
    nn.quantize(inmem, class_predicate=lambda p, m: qcfg.get(p) if isinstance(qcfg.get(p), dict) else False)
    assert mx.allclose(logits(inmem), mixed, atol=1e-5, rtol=1e-5).item()

    def rel(x):
        return (mx.linalg.norm((x - ref).reshape(-1)) / mx.linalg.norm(ref.reshape(-1))).item()

    assert rel(logits(load_model(q8_dir))) < rel(mixed)


@pytest.mark.parametrize("family", FAMILIES)
def test_lsq_rounding_same_layout_lower_error_and_uses_imatrix(tiny, tmp_path, family):
    import shutil

    from tests.fixtures.build_tiny import CONFIGS

    t = tiny(family)
    calib = tmp_path / "calib"
    shutil.copytree(t["calib"], calib)
    tc = CONFIGS[family]["text_config"]
    E = tc["num_experts" if family == "qwen" else "n_routed_experts"]
    rng = np.random.default_rng(0)
    for i in range(tc["num_hidden_layers"]):  # calib-style per-expert input second moments
        np.savez(calib / f"layer_{i:03d}.npz",
                 **{"act_sqmean.mlp.switch_mlp.gate_proj": rng.gamma(1.0, 1.0, (E, tc["hidden_size"])).astype(np.float32)})
    common = ["--model", family, "--calib", calib, "--prune", 0.5, "--mtp", "drop", "--budget-gb", 1,
              "--pin", r"switch_mlp\.=3:32", "--pin", r"(experts|attn)\.=4:32"]
    rtn = run(t["src"], tmp_path / "rtn", *common)
    lsq = run(t["src"], tmp_path / "lsq", *common, "--rounding", "lsq")
    a, b = load_dir(tmp_path / "rtn"), load_dir(tmp_path / "lsq")
    assert set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    assert lsq["written_bytes"] == rtn["written_bytes"]
    r = lsq["rounding"]
    assert r["method"] == "lsq"
    stats = [v for k, v in r.items() if isinstance(v, dict)]
    assert stats and all(v["err_lsq"] <= v["err_rtn"] * (1 + 1e-6) for v in stats)
    assert sum(v["err_lsq"] for v in stats) < 0.95 * sum(v["err_rtn"] for v in stats)
    if family == "glm":  # glm's switch_mlp gate_proj is a stacked [E, out, in] tensor at write time
        assert any(v["weighted"] for k, v in r.items() if isinstance(v, dict) and k.startswith("expert"))
    generate_tokens(tmp_path / "lsq")
