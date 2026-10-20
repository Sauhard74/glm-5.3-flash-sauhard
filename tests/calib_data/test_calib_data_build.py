import json

import pytest

from calib_data import build, sources
from calib_data_helpers import ROOT, tokenizer_dir

DOMAINS = {"code", "agentic", "reasoning", "longctx"}


def run(model, out, *extra):
    build.main(["--model", model, "--out", str(out), "--tokenizer", str(tokenizer_dir(model)), *extra])
    rows = {split: [json.loads(line) for line in (out / f"{split}.jsonl").read_text().splitlines()]
            for split in ("calib", "eval")}
    return rows, json.loads((out / "manifest.json").read_text())


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_smoke_contract(tmp_path, model, renderers):
    rows, man = run(model, tmp_path / "a", "--smoke", "--preview")
    vocab = renderers(model).vocab_size
    assert len(rows["calib"]) == 8 and len(rows["eval"]) == 4
    for split, rs in rows.items():
        assert {r["domain"] for r in rs} == DOMAINS
        for r in rs:
            assert {"id", "domain", "tokens"} <= set(r)
            assert r["tokens"] and all(isinstance(t, int) and 0 <= t < vocab for t in r["tokens"])
    longs = [r for r in rows["calib"] + rows["eval"] if r["domain"] == "longctx"]
    assert all(int(2048 * 0.95) <= len(r["tokens"]) <= 4096 for r in longs)
    assert not {r["src"] for r in rows["calib"]} & {r["src"] for r in rows["eval"]}
    assert man["model"] == model and man["stats"]["calib"]["n"] == 8
    assert (tmp_path / "a" / "preview").is_dir()
    bos = renderers(model).tok.convert_tokens_to_ids("[gMASK]" if model == "glm" else "<|im_start|>")
    assert all(r["tokens"][0] == bos for r in rows["calib"])


def test_deterministic_and_seeded(tmp_path):
    a, _ = run("qwen", tmp_path / "a", "--smoke")
    b, _ = run("qwen", tmp_path / "b", "--smoke")
    c, _ = run("qwen", tmp_path / "c", "--smoke", "--seed", "1")
    assert a == b
    assert a["calib"] != c["calib"]
    assert {r["src"] for r in a["eval"]} & {r["src"] for r in c["calib"]} == set()


def test_mix_and_counts(tmp_path):
    rows, man = run("glm", tmp_path / "m", "--source", "fixtures", "--samples", "10", "--eval-samples", "3",
                    "--mix", "code=1,agentic=1,reasoning=0,longctx=0", "--eval-mix", "reasoning=1",
                    "--max-len", "code=512,agentic=1024,reasoning=1024", "--longctx", "2048-4096")
    assert [r["domain"] for r in rows["calib"]].count("code") == 5
    assert {r["domain"] for r in rows["calib"]} == {"code", "agentic"}
    assert {r["domain"] for r in rows["eval"]} == {"reasoning"} and len(rows["eval"]) == 3
    assert max(len(r["tokens"]) for r in rows["calib"] if r["domain"] == "code") <= 512
    assert man["mix"]["calib"] == {"code": 5, "agentic": 5, "reasoning": 0, "longctx": 0}


def test_unknown_domain_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run("glm", tmp_path / "x", "--smoke", "--mix", "poetry=1")


def _cached():
    cache = ROOT / "artifacts" / "cache" / "calib_data"
    return all(sources._cache_path(cache, p).exists() for parts in sources.SOURCES.values() for p in parts)


@pytest.mark.skipif(not _cached(), reason="real source slices not cached (run calib_data.build once)")
@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_real_sources_offline(tmp_path, model):
    rows, man = run(model, tmp_path / "r", "--offline", "--cache-dir", str(ROOT / "artifacts/cache/calib_data"),
                    "--samples", "24", "--eval-samples", "8", "--longctx", "16384-24576")
    by_src = man["stats"]["calib"]["by_source"]
    assert set(by_src) >= {"gh_files", "commit_edit", "openhands", "toucan_multi", "openr1_math", "openthoughts_science"}
    assert {r["domain"] for r in rows["calib"]} == DOMAINS
    for r in rows["calib"] + rows["eval"]:
        if r["domain"] == "longctx":
            assert 15_000 <= len(r["tokens"]) <= 24_576
        else:
            assert len(r["tokens"]) <= build.MAX_LEN[r["domain"]]
    assert man["sources"]["openhands"][0]["rev"] == sources.REV["SWE-Gym/OpenHands-SFT-Trajectories"]
