import json

import pytest

from evals import code, compare, kl, longctx, math, speed, tools


def _run(module, model, results, *extra):
    module.main(["--model", str(model), "--smoke", "--results-dir", str(results), "--max-weights-gb", "1", *extra])
    with open(results / model.name / f"{module.__name__.split('.')[-1]}.json") as f:
        return json.load(f)


def _check_schema(rec, suite):
    for key in ("suite", "model", "model_path", "family", "settings", "summary", "breakdown", "samples", "runtime"):
        assert key in rec
    assert rec["suite"] == suite and rec["smoke"] is True
    assert rec["summary"]["headline"] in rec["summary"]


@pytest.mark.parametrize("family", ["qwen", "glm"])
def test_all_suites_smoke(tiny_models, family, tmp_path):
    model = tiny_models[family]
    results = tmp_path / "results"
    for module in (code, math, tools, longctx):
        rec = _run(module, model, results)
        _check_schema(rec, module.__name__.split(".")[-1])
        assert rec["family"] == family
        assert rec["settings"]["mtp"] == "on"
        assert rec["samples"], module.__name__
        for s in rec["samples"]:
            assert s["generation_tokens"] > 0
            assert s["thinking_closed"]
    lc = json.loads((results / model.name / "longctx.json").read_text())
    for s in lc["samples"]:
        assert abs(s["prompt_tokens"] - s["length"]) <= max(16, 0.02 * s["length"])

    rec = _run(kl, model, results)
    _check_schema(rec, "kl")
    assert rec["summary"]["mean_kl"] < 1e-4
    assert set(rec["breakdown"]) == {"code", "reasoning", "longctx"}

    rec = _run(speed, model, results)
    _check_schema(rec, "speed")
    assert {r["mtp"] for r in rec["samples"]} == {"off", "on"}
    for r in rec["samples"]:
        assert r["decode_tokens"] == 16 and r["prefill_tps"] > 0 and r["peak_mlx_gb"] > 0

    md = compare.render(compare.load_results(results, None, include_smoke=True))
    assert f"| {model.name} |" in md and "## speed" in md and "## tools" in md


def test_mtp_off_and_no_thinking(tiny_models, tmp_path):
    rec = _run(math, tiny_models["glm"], tmp_path, "--mtp", "off", "--no-thinking", "--datasets", "math500")
    assert rec["settings"]["mtp"] == "off" and rec["settings"]["thinking"] is False
    assert all(s["reasoning"] == "" for s in rec["samples"])


def test_oracle_modes(tiny_models, tmp_path):
    model = tiny_models["qwen"]
    assert _run(code, model, tmp_path, "--oracle")["summary"]["pass@1"] == 1.0
    assert _run(math, model, tmp_path, "--oracle")["summary"]["accuracy"] == 1.0
    assert _run(tools, model, tmp_path, "--oracle")["summary"]["accuracy"] == 1.0


def test_weight_guard(tiny_models, tmp_path):
    with pytest.raises(SystemExit, match="refusing to load"):
        math.main(["--model", str(tiny_models["qwen"]), "--smoke", "--results-dir", str(tmp_path), "--max-weights-gb", "0.001"])


@pytest.mark.parametrize("family", ["qwen", "glm"])
def test_generator_render_and_budget(tiny_models, family):
    from mlx_vlm import load

    from evals.generation import Generator, Sampling

    model, processor = load(str(tiny_models[family]))
    gen = Generator(model, processor, tiny_models[family], thinking=False, answer_tokens=8, sampling=Sampling())
    prompt, ids = gen.render([{"role": "user", "content": "hi"}])
    assert prompt.rstrip().endswith("</think>") or prompt.endswith("</think>\n\n")
    assert ids == gen.tokenizer.encode(prompt, add_special_tokens=False)

    gen = Generator(model, processor, tiny_models[family], thinking=True, thinking_budget=6, answer_tokens=5, sampling=Sampling())
    prompt, _ = gen.render([{"role": "user", "content": "hi"}])
    assert prompt.rstrip().endswith("<think>")
    out = gen.generate([{"role": "user", "content": "hi"}])
    assert out.thinking_closed
    if out.extra["budget_forced"]:
        assert out.generation_tokens <= 6 + 5
        assert out.extra["passes"] == 2


def test_resume_skips_finished_items(tiny_models, tmp_path, monkeypatch):
    from evals.generation import Generator

    model = tiny_models["qwen"]
    real = Generator.generate
    calls = {"n": 0, "fail_at": 3}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == calls["fail_at"]:
            raise RuntimeError("boom")
        return real(self, *a, **k)

    monkeypatch.setattr(Generator, "generate", flaky)
    argv = ["--model", str(model), "--smoke", "--results-dir", str(tmp_path), "--mtp", "off"]
    with pytest.raises(RuntimeError):
        math.main(argv)
    partial = tmp_path / model.name / "math.partial.jsonl"
    assert partial.exists()
    calls.update(n=0, fail_at=-1)
    math.main(argv + ["--resume"])
    assert calls["n"] == 2
    assert not partial.exists()
    rec = json.loads((tmp_path / model.name / "math.json").read_text())
    assert len(rec["samples"]) == 4
