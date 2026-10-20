import json
import random

import pytest

from calib_data import fixtures, recipes
from calib_data.build import allocate, parse_kv, pool_for, split_of
from calib_data.recipes import RECIPES


def test_allocate_sums_and_floors():
    assert allocate(512, {"code": 192, "agentic": 160, "reasoning": 144, "longctx": 16}) == \
        {"code": 192, "agentic": 160, "reasoning": 144, "longctx": 16}
    a = allocate(10, {"a": 0.97, "b": 0.01, "c": 0.02})
    assert sum(a.values()) == 10 and min(a.values()) >= 1
    assert allocate(1, {"a": 1, "b": 1, "c": 1}) in ({"a": 1, "b": 0, "c": 0}, {"a": 0, "b": 1, "c": 0},
                                                     {"a": 0, "b": 0, "c": 1})
    assert sum(allocate(1, {"a": 1, "b": 1, "c": 1}).values()) == 1
    assert allocate(5, {"a": 1, "b": 0}) == {"a": 5, "b": 0}


def test_parse_kv():
    assert parse_kv("code=0.5,agentic=1") == {"code": 0.5, "agentic": 1.0}
    assert parse_kv("code=4096", int) == {"code": 4096}


def test_think_split():
    assert recipes.think_split("<think>\nabc\n</think>\n\nfinal") == ("abc", "final")
    assert recipes.think_split("no tags") == (None, "no tags")


def test_split_is_stable_and_disjoint():
    rec = next(r for r in RECIPES if r.name == "openr1_math")
    rows = fixtures.fixture_rows("openr1_math") * 1
    first = [split_of(rec, r) for r in rows]
    assert first == [split_of(rec, r) for r in rows]
    big = [{"_src": f"x:{i}"} for i in range(5000)]
    frac = sum(split_of(rec, r) == "eval" for r in big) / len(big)
    assert 0.25 < frac < 0.35
    calib = {r["_src"] for r in pool_for(rec, big, "calib", 0)}
    evals = {r["_src"] for r in pool_for(rec, big, "eval", 0)}
    assert not calib & evals and len(calib | evals) == len(big)
    assert [r["_src"] for r in pool_for(rec, big, "calib", 0)] == [r["_src"] for r in pool_for(rec, big, "calib", 0)]
    assert [r["_src"] for r in pool_for(rec, big, "calib", 0)] != [r["_src"] for r in pool_for(rec, big, "calib", 1)]


def test_shards_partition_github_code():
    a, b = (next(r for r in RECIPES if r.name == n) for n in ("gh_files", "code_repo"))
    rows = [{"_src": f"g:{i}"} for i in range(1000)]
    pa = {r["_src"] for r in pool_for(a, rows, "calib", 0)}
    pb = {r["_src"] for r in pool_for(b, rows, "calib", 0)}
    assert pa and pb and not pa & pb


def test_openhands_tools_and_messages():
    raw = fixtures.fixture_rows("openhands")[0]["messages"]
    msgs, tools = recipes.openhands_messages(raw)
    names = [t["function"]["name"] for t in tools]
    assert names == ["execute_bash", "str_replace_editor"]
    ed = tools[1]["function"]["parameters"]
    assert ed["required"] == ["command", "path"]
    assert ed["properties"]["command"]["enum"] == ["view", "create", "str_replace"]
    assert ed["properties"]["view_range"]["type"] == "array"
    assert msgs[0]["role"] == "system" and "following functions" not in msgs[0]["content"]
    calls = [tc for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    assert calls and all(isinstance(tc["function"]["arguments"], dict) for tc in calls)
    assert calls[-1]["function"]["arguments"]["view_range"] == [1, 20]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == len(calls)
    assert all(not m["content"].startswith("EXECUTION RESULT") for m in tool_msgs)
    assert {m["tool_call_id"] for m in tool_msgs} == {tc["id"] for tc in calls}


def test_toucan_merges_assistant_turns():
    row = fixtures.fixture_rows("toucan_multi")[0]
    msgs = recipes.toucan_messages(json.loads(row["messages"]))
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"] * 2
    a = msgs[1]
    assert a["content"] and len(a["tool_calls"]) == 1
    assert isinstance(a["tool_calls"][0]["function"]["arguments"], dict)
    assert msgs[2]["tool_call_id"] == a["tool_calls"][0]["id"]


def test_toucan_parallel_calls():
    raw = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": "", "function_call": {"name": "f", "arguments": "{\"a\": 1}"}},
           {"role": "assistant", "content": "", "function_call": {"name": "g", "arguments": "{}"}},
           {"role": "function", "name": "f", "content": "1"},
           {"role": "function", "name": "g", "content": "2"},
           {"role": "assistant", "content": "done"}]
    msgs = recipes.toucan_messages(raw)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [tc["function"]["name"] for tc in msgs[1]["tool_calls"]] == ["f", "g"]
    assert [m["tool_call_id"] for m in msgs[2:4]] == ["call_0", "call_1"]


def test_clean_story_strips_html_and_gutenberg():
    movie = "<html><head><script>var x=1;</script></head><body>MENU<pre>\nINT. ROOM - NIGHT\n&amp; stuff\n</pre>footer</body></html>"
    out = recipes.clean_story(movie)
    assert "INT. ROOM" in out and "& stuff" in out and "var x" not in out and "MENU" not in out and "footer" not in out
    book = "ï»¿The Project Gutenberg EBook\n*** START OF THIS BOOK ***\nChapter 1\n" + "text " * 2000 + "\n*** END OF THIS BOOK ***\nlicense"
    out = recipes.clean_story(book)
    assert out.startswith("Chapter 1") and "license" not in out


def test_code_filters():
    assert recipes.code_ok("x = 1\n" * 100)
    assert not recipes.code_ok("x" * 2000)
    assert not recipes.code_ok("// Code generated by protoc. DO NOT EDIT.\n" + "x = 1\n" * 100)


def test_clip_output():
    assert recipes.clip_output("a" * 10, 100) == "a" * 10
    out = recipes.clip_output("a" * 1000 + "b" * 1000, 100)
    assert out.startswith("a" * 50) and out.endswith("b" * 50) and "truncated" in out


def test_loguniform_bounds():
    rng = random.Random(0)
    xs = [recipes.loguniform(rng, 100, 1000) for _ in range(1000)]
    assert min(xs) >= 100 and max(xs) <= 1000


@pytest.mark.parametrize("name", sorted(fixtures.BUILDERS))
def test_fixture_rows_cover_sources(name):
    from calib_data.sources import SOURCES

    assert name in SOURCES
    rows = fixtures.fixture_rows(name)
    assert rows and all("_src" in r for r in rows)
    assert rows == fixtures.fixture_rows(name)


def test_every_recipe_has_fixture_source():
    assert {r.source for r in RECIPES} <= set(fixtures.BUILDERS)
