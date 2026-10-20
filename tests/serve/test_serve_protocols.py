import json

import pytest

from serve_helpers import ARCHS, sse_events

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
TOOLS_ANTHROPIC = [
    {
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": TOOLS_OPENAI[0]["function"]["parameters"],
    }
]
NATIVE_TOOL_CALL = {
    "qwen": "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
    "glm": "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>",
}


@pytest.mark.parametrize("arch", ARCHS)
def test_models_and_chat(arch, tiny_models, serve_client):
    c = serve_client(tiny_models[arch], served_name="local-coder")
    models = c.get("/v1/models").json()["data"]
    assert [m["id"] for m in models] == ["local-coder"] and models[0]["loaded"]

    r = c.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "gpt-4o"
    assert 0 < body["usage"]["completion_tokens"] <= 5

    r = c.post(
        "/v1/chat/completions",
        json={"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert r.status_code == 200
    chunks = sse_events(r.text)
    assert chunks and chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop")


@pytest.mark.parametrize("arch", ARCHS)
def test_openai_tool_calls(arch, tiny_models, serve_client, force_output):
    c = serve_client(tiny_models[arch], mtp="off")
    force_output(c, NATIVE_TOOL_CALL[arch])
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": TOOLS_OPENAI,
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    calls = choice["message"]["tool_calls"]
    assert choice["finish_reason"] == "tool_calls"
    assert calls[0]["function"]["name"] == "Bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}

    force_output(c, NATIVE_TOOL_CALL[arch])
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "stream": True,
            "messages": [{"role": "user", "content": "list files"}],
            "tools": TOOLS_OPENAI,
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    deltas = [ch["choices"][0]["delta"] for ch in sse_events(r.text) if ch.get("choices")]
    streamed = [tc for d in deltas for tc in (d.get("tool_calls") or [])]
    assert streamed and streamed[0]["function"]["name"] == "Bash"
    args = "".join(tc["function"].get("arguments") or "" for tc in streamed)
    assert json.loads(args) == {"command": "ls -la"}


@pytest.mark.parametrize("arch", ARCHS)
def test_anthropic_tool_use(arch, tiny_models, serve_client, force_output):
    c = serve_client(tiny_models[arch], mtp="off")
    request = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "temperature": 0,
        "system": [{"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "list files"}]}],
        "tools": TOOLS_ANTHROPIC,
    }
    force_output(c, NATIVE_TOOL_CALL[arch])
    r = c.post("/v1/messages", json=request, headers={"anthropic-version": "2023-06-01"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stop_reason"] == "tool_use"
    tool = next(b for b in body["content"] if b["type"] == "tool_use")
    assert tool["name"] == "Bash" and tool["input"] == {"command": "ls -la"}
    assert tool["id"]

    force_output(c, NATIVE_TOOL_CALL[arch])
    r = c.post("/v1/messages?beta=true", json={**request, "stream": True})
    assert r.status_code == 200
    events = sse_events(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "message_start" and types[-1] == "message_stop"
    start = next(e for e in events if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use")
    assert start["content_block"]["name"] == "Bash"
    partial = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(partial) == {"command": "ls -la"}
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_anthropic_thinking_block(tiny_models, serve_client, force_output):
    c = serve_client(tiny_models["qwen"], mtp="off")
    force_output(c, "<think>\nplan the change\n</think>\n\nDone.")
    r = c.post(
        "/v1/messages",
        json={
            "model": "claude",
            "max_tokens": 64,
            "temperature": 0,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "go"}],
        },
    )
    assert r.status_code == 200, r.text
    blocks = {b["type"]: b for b in r.json()["content"]}
    assert "plan the change" in blocks["thinking"]["thinking"]
    assert blocks["text"]["text"].strip() == "Done."


def test_count_tokens(tiny_models, serve_client):
    c = serve_client(tiny_models["glm"])
    r = c.post(
        "/v1/messages/count_tokens",
        json={"model": "claude", "messages": [{"role": "user", "content": "hello there"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["input_tokens"] > 2


def test_api_key(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"], api_key="sekrit", preload=False)
    assert c.get("/v1/models").status_code == 401
    assert c.get("/serve/status").status_code == 401
    msg = {"model": "x", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/messages", json=msg).status_code == 401
    assert not c.app.state.engine.is_loaded()  # unauthenticated traffic never loads the model
    auth = {"Authorization": "Bearer sekrit"}
    assert c.get("/v1/models", headers=auth).status_code == 200
    assert c.post("/v1/messages", json=msg, headers=auth).status_code == 200
    assert c.post("/v1/messages", json=msg, headers={"x-api-key": "sekrit"}).status_code == 200
    assert c.post("/v1/messages", json=msg, headers={"x-api-key": "wrong"}).status_code == 401


def test_settings_mutation_removed(tiny_models, serve_client):
    c = serve_client(tiny_models["qwen"])
    r = c.patch("/v1/settings", json={"draft_model": "some/other-model"})
    assert r.status_code in (404, 405)
