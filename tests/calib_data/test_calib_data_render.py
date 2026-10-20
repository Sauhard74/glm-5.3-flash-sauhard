import pytest

from calib_data.recipes import assistant, tool_call

MSGS = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "List the repo."},
    assistant("Let me look.", tool_calls=[tool_call(0, "execute_bash", {"command": "ls -la"})]),
    {"role": "tool", "tool_call_id": "call_0", "content": "README.md\nsrc"},
    assistant("There is a README and a src dir.", reasoning="The listing shows two entries."),
]
TOOLS = [{"type": "function", "function": {
    "name": "execute_bash", "description": "Run a command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]

NATIVE = {
    "qwen": ["<|im_start|>system\n", "<tools>\n{\"type\": \"function\"",
             "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
             "<|im_start|>user\n<tool_response>\nREADME.md\nsrc\n</tool_response><|im_end|>",
             "<think>\nThe listing shows two entries.\n</think>\n\nThere is a README"],
    "glm": ["[gMASK]<sop>", "<tools>\n{\"name\": \"execute_bash\"",
            "<tool_call>execute_bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>",
            "<|observation|><tool_response>README.md\nsrc</tool_response>",
            "<|assistant|><think>The listing shows two entries.</think>There is a README"],
}
SPECIAL = {"qwen": ["<|im_start|>", "<|im_end|>"], "glm": ["[gMASK]", "<sop>", "<|user|>", "<|assistant|>", "<|observation|>"]}


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_native_tool_format(renderers, model):
    r = renderers(model)
    text = r.text(MSGS, TOOLS)
    for frag in NATIVE[model]:
        assert frag in text, (model, frag)


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_special_tokens_are_single_ids(renderers, model):
    r = renderers(model)
    ids = r.render(MSGS, TOOLS)
    for s in SPECIAL[model]:
        tid = r.tok.convert_tokens_to_ids(s)
        assert r.encode(s) == [tid]
        assert tid in ids
    assert max(ids) < r.vocab_size
    assert r.tok.decode(ids) == r.text(MSGS, TOOLS)


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_generation_prompt_opens_thinking(renderers, model):
    r = renderers(model)
    text = r.text(MSGS[:2], TOOLS, gen=True)
    assert text.endswith("<|assistant|><think>" if model == "glm" else "<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_fit_drops_trailing_turns(renderers, model):
    r = renderers(model)
    long = MSGS[:2] + [assistant("x " * 50, tool_calls=[tool_call(i, "execute_bash", {"command": f"echo {i}"})])
                       if k == 0 else {"role": "tool", "tool_call_id": f"call_{i}", "content": "ok " * 50}
                       for i in range(20) for k in range(2)]
    full = r.render(long, TOOLS, gen=True)
    cap = len(full) // 2
    ids = r.fit(long, TOOLS, cap)
    assert 0 < len(ids) <= cap
    assert full[: len(ids) - 8] == ids[: len(ids) - 8]
    assert r.fit(long, TOOLS, len(full) + 10) == full
    assert len(r.fit(MSGS[:2] + [assistant("y " * 5000)], None, 300)) == 300


@pytest.mark.parametrize("model", ["glm", "qwen"])
def test_clip_and_count(renderers, model):
    r = renderers(model)
    text = "def f(x):\n    return x + 1\n" * 200
    clipped = r.clip(text, 50)
    assert text.startswith(clipped[:-5]) and r.count(clipped) <= 52
    assert r.clip("short", 50) == "short"
