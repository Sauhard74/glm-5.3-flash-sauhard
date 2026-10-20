import shutil
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

from serve import memory
from serve.config import GB, ServeConfig, model_fingerprint
from serve_helpers import ARCHS, sse_events

SYSTEM = "You are a coding agent. " + " ".join(f"Rule {i}: keep code tidy and tested." for i in range(120))


def agent_turns(n=3):
    """Successive Anthropic requests of one agent loop (each extends the last)."""
    messages = [{"role": "user", "content": "list the files"}]
    for i in range(n):
        yield {"model": "claude", "max_tokens": 6, "system": SYSTEM, "messages": list(messages)}
        messages += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_{i}", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": f"file_{i}.py"}]},
        ]


TOOLS = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]


def test_weights_and_refusal(tiny_models, tmp_path):
    model = tiny_models["qwen"]
    nbytes = memory.weights_bytes(model)
    assert nbytes > 1 << 20
    cfg = ServeConfig(model=model, memory_limit_gb=nbytes / GB * 0.5, kv_reserve_gb=0)
    with pytest.raises(memory.ModelTooLarge, match="exceeds"):
        memory.check_fits(cfg)
    cfg = ServeConfig(model=model, memory_limit_gb=1.0, kv_reserve_gb=0.99)
    with pytest.raises(memory.ModelTooLarge):
        memory.check_fits(cfg)
    assert memory.check_fits(ServeConfig(model=model, memory_limit_gb=1.0, kv_reserve_gb=0.5)) == nbytes


def test_cli_refuses_oversized_model(tiny_models):
    from serve.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--model", tiny_models["glm"], "--memory-limit-gb", "0.01", "--kv-reserve-gb", "0"])
    assert exc.value.code == 2


def test_lazy_load_refused_over_cap(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"], preload=False, memory_limit_gb=0.01, kv_reserve_gb=0)
    assert not c.get("/serve/status").json()["loaded"]
    r = c.post("/v1/messages", json={"model": "x", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 507
    assert "memory cap" in r.json()["error"]["message"]
    assert not c.get("/serve/status").json()["loaded"]


def _current_wired_limit():
    old = mx.set_wired_limit(0)
    mx.set_wired_limit(old)
    return old


def test_limits_applied(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"], memory_limit_gb=12, cache_limit_gb=1, kv_reserve_gb=2)
    st = c.get("/serve/status").json()
    assert mx.get_memory_limit() == 12 * GB
    assert st["memory"]["cap_gb"] == 12
    assert 2.0 < st["memory"]["wired_limit_gb"] <= 12
    assert st["prefix_cache"]["memory_max_bytes"] <= 12 * GB
    wired = c.app.state.engine.limits["wired_limit"]
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
    assert r.status_code == 200
    # The batch generator stays alive between requests; it must not have raised the limit.
    assert _current_wired_limit() == wired
    c.post("/serve/unload")
    assert _current_wired_limit() == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_prefix_cache_across_agent_turns(arch, tiny_models, serve_client):
    c = serve_client(tiny_models[arch])
    usages = []
    for req in agent_turns(3):
        r = c.post("/v1/messages", json={**req, "tools": TOOLS})
        assert r.status_code == 200, r.text
        usages.append(r.json()["usage"])
    assert usages[0]["cache_read_input_tokens"] == 0
    for prev, cur in zip(usages, usages[1:]):
        # Everything the previous turn prefilled is reused (minus the guard token).
        assert cur["cache_read_input_tokens"] >= prev["input_tokens"] + prev["cache_read_input_tokens"] - 2
        assert cur["input_tokens"] < 100
    apc = c.get("/serve/status").json()["prefix_cache"]
    assert apc["exact_hits"] >= 2 and apc["rejects"] == 0


def test_prefix_cache_survives_unload(tiny_models, serve_client):
    c = serve_client(tiny_models["glm"])
    turns = list(agent_turns(2))
    assert c.post("/v1/messages", json=turns[0]).status_code == 200
    time.sleep(0.5)  # let the SSD writer drain
    assert c.post("/serve/unload").json()["status"] == "unloaded"
    assert not c.get("/serve/status").json()["loaded"]
    r = c.post("/v1/messages", json=turns[1])
    assert r.status_code == 200
    assert r.json()["usage"]["cache_read_input_tokens"] > 1000
    st = c.get("/serve/status").json()
    assert st["load_count"] == 2 and st["prefix_cache"]["disk_hits"] >= 1


def test_prefix_cache_keyed_by_checkpoint(tiny_models, tmp_path):
    a = ServeConfig(model=tiny_models["qwen"], prefix_cache_dir=str(tmp_path))
    b = ServeConfig(model=tiny_models["glm"], prefix_cache_dir=str(tmp_path))
    assert a.prefix_cache_path() != b.prefix_cache_path()
    copy = shutil.copytree(tiny_models["qwen"], tmp_path / "rebuilt")
    before = model_fingerprint(str(copy))
    with open(copy / "config.json", "a") as f:
        f.write("\n")
    assert model_fingerprint(str(copy)) != before

    from serve.engine import Engine

    cfg = ServeConfig(model=str(copy), prefix_cache_dir=str(tmp_path / "pc"))
    current = Path(cfg.prefix_cache_path())
    stale = current.with_name(f"{cfg.name}-{'0' * 12}")
    other = current.with_name(f"another-model-{'0' * 12}")
    for d in (current, stale, other):
        d.mkdir(parents=True)
    Engine(cfg)._prune_stale_prefix_caches()
    assert current.exists() and other.exists() and not stale.exists()


@pytest.mark.parametrize("arch", ARCHS)
def test_mtp_speculative(arch, tiny_models, serve_client):
    c = serve_client(tiny_models[arch])
    spec = c.get("/serve/status").json()["speculative"]
    assert spec["active"] and spec["kind"] == "mtp"
    r = c.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 0},
    )
    assert r.status_code == 200, r.text
    timings = r.json()["timings"]
    assert timings["draft_kind"] == "mtp" and timings["draft_rounds"] > 0
    # Claude Code sends thinking budgets; mlx-vlm can't enforce them while drafting.
    r = c.post(
        "/v1/messages",
        json={
            "model": "claude",
            "max_tokens": 8,
            "thinking": {"type": "enabled", "budget_tokens": 4000},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text


def test_mtp_off(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"], mtp="off")
    assert c.get("/serve/status").json()["speculative"]["active"] is False
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
    assert r.json()["timings"].get("draft_kind") is None


def test_mtp_on_requires_head(tiny_models, tmp_path):
    from serve.engine import resolve_draft_model
    from serve.testing import build_tiny

    plain = build_tiny("qwen", tmp_path / "plain", mtp=False)
    assert resolve_draft_model(ServeConfig(model=str(plain))) is None
    with pytest.raises(ValueError, match="no MTP head"):
        resolve_draft_model(ServeConfig(model=str(plain), mtp="on"))
    assert resolve_draft_model(ServeConfig(model=tiny_models["glm"])) == tiny_models["glm"]


def test_context_and_output_clamps(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"], max_context=160, max_output_tokens=40, mtp="off")
    msg = [{"role": "user", "content": " ".join(["word"] * 100)}]
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": msg, "max_tokens": 32000})
    assert r.status_code == 200, r.text
    usage = r.json()["usage"]
    assert usage["completion_tokens"] <= min(40, 160 - usage["prompt_tokens"])
    r = c.post("/v1/messages", json={"model": "x", "max_tokens": 32000, "stream": True, "messages": msg})
    assert r.status_code == 200 and "message_stop" in r.text
    too_long = [{"role": "user", "content": " ".join(["word"] * 400)}]
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": too_long, "max_tokens": 10})
    assert r.status_code == 400


def test_single_sequence_queue(tiny_models, serve_client, monkeypatch):
    from serve import engine as engine_mod

    c = serve_client(tiny_models["qwen"], mtp="off")
    seen = []
    orig = engine_mod._ORIG["step"]

    def recording(self_, batch_gen, active, gen_kwargs=None):
        seen.append(len(active))
        return orig(self_, batch_gen, active, gen_kwargs)

    monkeypatch.setitem(engine_mod._ORIG, "step", recording)
    results = []

    def call(i):
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": f"req {i}"}], "max_tokens": 12},
        )
        results.append(r.status_code)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200, 200, 200]
    assert seen and max(seen) == 1


def test_memory_guard_fails_request_not_server(tiny_models, serve_client):
    c = serve_client(tiny_models["glm"])
    engine = c.app.state.engine
    msg = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    real = engine.guard.limit_bytes
    engine.guard.limit_bytes = 1
    try:
        r = c.post("/v1/chat/completions", json=msg)
        assert r.status_code >= 400 and "Memory cap exceeded" in r.text
    finally:
        engine.guard.limit_bytes = real
    assert c.get("/serve/status").json()["memory"]["guard_trips"] >= 1
    assert c.post("/v1/chat/completions", json=msg).status_code == 200


def test_idle_unload_and_reload(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"], idle_unload_s=1)
    loaded_bytes = mx.get_active_memory()
    assert c.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}).status_code == 200
    deadline = time.time() + 10
    while time.time() < deadline and c.get("/serve/status").json()["loaded"]:
        time.sleep(0.25)
    st = c.get("/serve/status").json()
    assert not st["loaded"] and st["unload_count"] == 1 and st["last_unload_reason"].startswith("idle")
    # Relative: pytest's log capture may pin tracebacks from earlier tests.
    assert loaded_bytes - mx.get_active_memory() >= 0.9 * memory.weights_bytes(tiny_models["qwen"])
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
    assert r.status_code == 200
    assert c.get("/serve/status").json()["load_count"] == 2


def test_unload_refused_while_busy(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"])
    engine = c.app.state.engine
    engine.begin()
    try:
        assert c.post("/serve/unload").status_code == 409
    finally:
        engine.end()
    assert c.post("/serve/unload").json()["status"] == "unloaded"


@pytest.mark.parametrize("arch", ARCHS)
def test_kv_quant_option(arch, tiny_models, serve_client):
    c = serve_client(tiny_models[arch], kv_bits=8)
    kv = c.get("/serve/status").json()["kv_quant"]
    assert kv["requested_bits"] == 8
    # Either mlx-vlm quantizes this architecture's caches, or the engine fell back
    # to full precision at load time with the reason recorded.
    assert kv["active_bits"] == 8 or (kv["active_bits"] is None and kv["error"])
    for req in agent_turns(2):
        r = c.post("/v1/messages", json=req)
        assert r.status_code == 200, r.text


def test_openai_agent_loop_reuses_prefix(tiny_models, serve_client):
    """OpenCode-style loop: assistant tool_calls + role=tool results, usage in the stream."""
    c = serve_client(tiny_models["qwen"])
    tools = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "list files"}]
    cached = []
    for i in range(3):
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": messages, "tools": tools, "max_tokens": 6, "stream": True,
                  "stream_options": {"include_usage": True}},
        )
        assert r.status_code == 200
        usage = [e["usage"] for e in sse_events(r.text) if e.get("usage")][-1]
        cached.append(usage["prompt_tokens_details"]["cached_tokens"])
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"file_{i}.py"},
        ]
    assert cached[0] == 0 and cached[1] > 900 and cached[2] > cached[1]


def test_glm_projected_prefill_cache_bounded(tiny_models, serve_client):
    c = serve_client(tiny_models["glm"], kv_reserve_gb=0.0001, prefill_step_size=64)
    st = c.get("/serve/status").json()
    # 2 DSA layers x 4 heads x (32 + 32) dims x 2 bytes = 1 KB/token; budget = reserve / 4.
    assert st["glm_projected_prefill_tokens"] == int(0.0001 * GB) // 4 // 1024
    usages = []
    for req in agent_turns(2):
        r = c.post("/v1/messages", json=req)
        assert r.status_code == 200, r.text
        usages.append(r.json()["usage"])
    assert usages[1]["cache_read_input_tokens"] > 500
