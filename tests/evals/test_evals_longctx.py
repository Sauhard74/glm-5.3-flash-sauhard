import random

import pytest

from evals.common import load_tokenizer
from evals.longctx import TASKS, ContextBuilder, make_task, score


@pytest.mark.parametrize("task", TASKS)
def test_tasks_embed_answers(task):
    t = make_task(task, random.Random(1), n_filler=200)
    joined = " ".join(text for _, text in t["inserts"])
    for a in t["answers"]:
        assert a in joined
    prompt = t["make_prompt"](ContextBuilder._assemble(t["filler"][:50], t["inserts"]))
    assert all(a in prompt for a in t["answers"])


def test_assemble_places_inserts_by_depth():
    units = [f"u{i}." for i in range(10)]
    out = ContextBuilder._assemble(units, [(0.0, "START"), (0.5, "MID"), (1.0, "END")])
    words = out.split()
    assert words[0] == "START" and words[-1] == "END"
    assert words.index("MID") == 6


@pytest.mark.parametrize("family", ["qwen", "glm"])
@pytest.mark.parametrize("target", [1024, 4096])
def test_builder_hits_target_length(tiny_models, family, target):
    tok = load_tokenizer(str(tiny_models[family]))

    def render(content):
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return text, tok.encode(text, add_special_tokens=False)

    for task in TASKS:
        t = make_task(task, random.Random(target), n_filler=max(400, target // 12))
        prompt, ids = ContextBuilder(tok, render).build(target, t["make_prompt"], t["inserts"], t["filler"])
        assert abs(len(ids) - target) <= max(16, 0.02 * target), (task, len(ids))
        assert all(a in prompt for a in t["answers"])


def test_score_is_recall():
    assert score(["ABCDE", "FGHIJ"], "the vars are abcde and something") == 0.5
    assert score(["1234567"], "1234567") == 1.0
