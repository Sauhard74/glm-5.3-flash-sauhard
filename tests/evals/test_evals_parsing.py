import pytest

from evals.parsing import (
    extract_code,
    extract_integer,
    extract_math_answer,
    last_boxed,
    math_equal,
    prompt_opens_thinking,
    split_reasoning,
)


def test_prompt_opens_thinking():
    assert prompt_opens_thinking("<|im_start|>assistant\n<think>\n")
    assert prompt_opens_thinking("<|assistant|><think>")
    assert not prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not prompt_opens_thinking("<|assistant|><think></think>")
    assert not prompt_opens_thinking("hello")


@pytest.mark.parametrize(
    "text,opened,expected",
    [
        ("reason here\n</think>\n\nThe answer<|im_end|>", True, ("reason here", "The answer", True)),
        ("still thinking...", True, ("still thinking...", "", False)),
        ("<think>r</think>a", False, ("r", "a", True)),
        ("plain answer<|endoftext|>", False, ("", "plain answer", True)),
        ("x</think>tool<|observation|>junk", True, ("x", "tool", True)),
    ],
)
def test_split_reasoning(text, opened, expected):
    assert split_reasoning(text, opened) == expected


def test_extract_code_prefers_last_python_block():
    text = "first\n```python\ndef a():\n    pass\n```\nthen\n```python\ndef b():\n    return 1\n```\n"
    assert extract_code(text) == "def b():\n    return 1"
    assert extract_code("```\nimport os\n```") == "import os"
    assert extract_code("```python\ndef unterminated():\n    return 2\n") .startswith("def unterminated")
    assert extract_code("def raw(): pass") == "def raw(): pass"


def test_last_boxed_balanced():
    assert last_boxed(r"so \boxed{\frac{1}{2}} and finally \boxed{x^{2}+1}") == "x^{2}+1"
    assert last_boxed(r"\boxed 5") == "5"
    assert last_boxed("nothing") is None


def test_extract_math_answer_fallbacks():
    assert extract_math_answer(r"The final answer is $42$.") == "42"
    assert extract_math_answer(r"... \boxed{204}") == "204"
    assert extract_integer("033") == 33
    assert extract_integer(r"\text{answer } 1{,}024") is not None


@pytest.mark.parametrize(
    "pred,gt",
    [
        (r"\frac{3}{4}", "0.75"),
        (r"\dfrac34", r"\frac{3}{4}"),
        ("3/4", r"\frac{3}{4}"),
        (r"(3,\frac{\pi}{2})", r"\left( 3, \frac{\pi}{2} \right)"),
        (r"10^\circ", "10"),
        (r"\$5.00", "5"),
        (r"x = 7", "7"),
        ("1,000", "1000"),
        (r"\text{(B)}", "(B)"),
        ("2, 1", "1, 2"),
    ],
)
def test_math_equal_positive(pred, gt):
    assert math_equal(pred, gt)


@pytest.mark.parametrize("pred,gt", [("3", "4"), (None, "1"), (r"\frac{1}{3}", "0.33"), ("(1,2)", "(2,1)")])
def test_math_equal_negative(pred, gt):
    assert not math_equal(pred, gt)
