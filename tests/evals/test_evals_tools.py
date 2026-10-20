import pytest

from evals.common import load_tokenizer
from evals.data import bfcl
from evals.tools import (
    check_item,
    convert_schema,
    get_tool_module,
    match_value,
    oracle_text,
    parse_calls,
    to_openai_tools,
)


def test_convert_schema_maps_bfcl_types():
    s = convert_schema(
        {
            "type": "dict",
            "properties": {
                "x": {"type": "float"},
                "t": {"type": "tuple", "items": {"type": "integer"}},
                "a": {"type": "any"},
                "o": {"type": "string", "optional": True},
            },
        }
    )
    assert s["type"] == "object"
    assert s["properties"]["x"]["type"] == "number"
    assert s["properties"]["t"] == {"type": "array", "items": {"type": "integer"}}
    assert "type" not in s["properties"]["a"]
    assert "optional" not in s["properties"]["o"]


@pytest.mark.parametrize(
    "value,expected,ok",
    [
        ("New York, NY", "new york ny", True),
        (10, 10.0, True),
        (10.5, 10, False),
        (True, True, True),
        ("true", True, True),
        (1, True, False),
        ([1, 2], [1, 2], True),
        ([2, 1], [1, 2], False),
        ({"a": 1}, {"a": [1], "b": [""]}, True),
        ({"a": 1, "c": 2}, {"a": [1]}, False),
        (None, None, True),
        ("null", None, True),
        ("2", 2, True),
    ],
)
def test_match_value(value, expected, ok):
    assert match_value(value, expected) is ok


def test_check_item_parallel_any_order_and_counts():
    item = bfcl("parallel", smoke=True)[0]
    calls = [("spotify.play", {"artist": "Maroon 5", "duration": 15}), ("spotify_play", {"artist": "taylor swift", "duration": 20})]
    assert check_item(item, calls)[0]
    assert not check_item(item, calls[:1])[0]
    assert not check_item(item, [calls[0], ("spotify.play", {"artist": "Maroon 5", "duration": 20})])[0]


def test_check_item_optional_and_unexpected_params():
    item = bfcl("simple", smoke=True)[0]
    assert check_item(item, [("calculate_triangle_area", {"base": 10, "height": 5})])[0]
    assert check_item(item, [("calculate_triangle_area", {"base": 10, "height": 5, "unit": "units"})])[0]
    assert not check_item(item, [("calculate_triangle_area", {"base": 10})])[0]
    assert not check_item(item, [("calculate_triangle_area", {"base": 10, "height": 5, "color": "red"})])[0]
    irr = bfcl("irrelevance", smoke=True)[0]
    assert check_item(irr, [])[0]
    assert not check_item(irr, [("determine_body_mass_index", {"weight": 1, "height": 2})])[0]


@pytest.mark.parametrize("family,parser", [("qwen", "qwen3_coder"), ("glm", "glm47")])
def test_native_format_round_trip(tiny_models, family, parser):
    tok = load_tokenizer(str(tiny_models[family]))
    name, module = get_tool_module(tok)
    assert name == parser
    for cat in ("simple", "multiple", "parallel", "irrelevance"):
        for item in bfcl(cat, smoke=True):
            tools = to_openai_tools(item["function"])
            text = oracle_text(tok, item, tools)
            ok, reason = check_item(item, parse_calls(text, module, tools))
            assert ok, (family, item["id"], reason, text)
