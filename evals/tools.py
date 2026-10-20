"""Function-calling accuracy on a BFCL v3 subset, rendered with the model's own
chat template (native tool format) and parsed with mlx-vlm's matching parser.

uv run python -m evals.tools --model models/<name> [--categories simple,parallel,...] [--limit 50]
"""

import json
import re
import time
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

from .common import Checkpoint, base_parser, load_model, load_tokenizer, mean, take, write_result
from .data import bfcl

DEFAULT_CATEGORIES = "simple,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple"
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "int": "integer", "bool": "boolean", "list": "array"}
_STR_STRIP = re.compile(r"[ \,\.\/\-\_\*\^]")
# Items whose ground truth contradicts their own schema (or that the native
# parsers cannot round-trip): the oracle fails them in the qwen3_coder and/or
# glm47 format, so they are excluded to keep families comparable.
INCONSISTENT_IDS = frozenset(
    """
    live_multiple_15-4-7 live_multiple_16-4-8 live_multiple_263-126-0 live_multiple_306-131-5
    live_multiple_338-133-2 live_multiple_342-133-6 live_multiple_343-133-7 live_multiple_344-133-8
    live_multiple_345-133-9 live_multiple_350-133-14 live_multiple_351-133-15 live_multiple_352-133-16
    live_multiple_354-133-18 live_multiple_495-148-5 live_multiple_500-148-10 live_multiple_507-149-4
    live_multiple_964-207-0 live_parallel_multiple_21-18-0 live_simple_106-63-0 live_simple_112-68-0
    live_simple_237-125-0 live_simple_238-125-1 live_simple_239-125-2 live_simple_240-125-3
    live_simple_241-125-4 parallel_multiple_21 parallel_multiple_75 simple_363
    """.split()
)


def convert_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [convert_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            if v == "any":
                continue
            out[k] = _TYPE_MAP.get(v, v)
        elif k in ("properties",) and isinstance(v, dict):
            out[k] = {name: convert_schema(s) for name, s in v.items()}
        elif k in ("items", "additionalProperties"):
            out[k] = convert_schema(v)
        elif k == "optional":
            continue
        else:
            out[k] = v
    return out


def to_openai_tools(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": convert_schema(f.get("parameters", {"type": "object", "properties": {}})),
            },
        }
        for f in functions
    ]


def norm_name(name: str) -> str:
    return name.strip().replace(".", "_")


def get_tool_module(tokenizer):
    from mlx_vlm.tools import _infer_tool_parser, load_tool_module

    name = _infer_tool_parser(getattr(tokenizer, "chat_template", None))
    if name is None:
        name = "json_tools"
    return name, load_tool_module(name)


def parse_calls(text: str, module, tools) -> List[Tuple[str, Dict[str, Any]]]:
    from mlx_vlm.tools import process_tool_calls

    calls = []
    for c in process_tool_calls(text, module, tools).calls:
        args = c["function"]["arguments"]
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            args = {"__raw__": args}
        calls.append((c["function"]["name"], args if isinstance(args, dict) else {"__raw__": args}))
    return calls


def _std(s: str) -> str:
    return _STR_STRIP.sub("", s).lower().replace("'", '"')


def _as_number(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def match_value(value: Any, expected: Any, ptype: Optional[str] = None) -> bool:
    # Native XML tool formats are stringly typed and BFCL answers do not always
    # agree with their own schemas, so scalars also match via their string form.
    if expected is None:
        return value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none"))
    if value is None:
        return isinstance(expected, str) and expected.strip().lower() in ("null", "none")
    if isinstance(expected, bool):
        if isinstance(value, str):
            return value.strip().lower() == str(expected).lower()
        return isinstance(value, bool) and value is expected
    if isinstance(value, bool):
        return isinstance(expected, str) and _std(expected) == str(value).lower()
    if isinstance(expected, str):
        if isinstance(value, (int, float)):
            value = str(value)
        return isinstance(value, str) and _std(value) == _std(expected)
    if isinstance(expected, (int, float)):
        value = _as_number(value)
        if not isinstance(value, (int, float)):
            return False
        if ptype in ("integer", "int") and isinstance(value, float) and not value.is_integer():
            return False
        return abs(float(value) - float(expected)) <= 1e-9 * max(1.0, abs(float(expected)))
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(value, (list, tuple))
            and len(value) == len(expected)
            and all(match_value(v, e) for v, e in zip(value, expected))
        )
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            return False
        for k, acc in expected.items():
            acc = acc if isinstance(acc, list) else [acc]
            if k not in value:
                if "" not in acc:
                    return False
            elif not any_match(value[k], acc):
                return False
        return all(k in expected for k in value)
    return value == expected


def any_match(value: Any, acceptable: List[Any], ptype: Optional[str] = None) -> bool:
    return any(match_value(value, e, ptype) for e in acceptable if not (isinstance(e, str) and e == ""))


def check_call(
    call: Tuple[str, Dict[str, Any]],
    expected: Dict[str, Dict[str, List[Any]]],
    docs: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Return None when the call matches the expected call, else a reason."""
    (gt_name, gt_params), = expected.items()
    name, args = call
    if norm_name(name) != norm_name(gt_name):
        return f"wrong function {name!r} (want {gt_name!r})"
    params = docs.get(norm_name(gt_name), {}).get("parameters", {})
    props = params.get("properties", {})
    for r in params.get("required", []):
        if r not in args:
            return f"missing required {r!r}"
    for k, v in args.items():
        if k not in gt_params:
            return f"unexpected parameter {k!r}"
        ptype = _TYPE_MAP.get(props.get(k, {}).get("type"), props.get(k, {}).get("type"))
        if not any_match(v, gt_params[k], ptype):
            return f"bad value for {k!r}: {v!r}"
    for k, acc in gt_params.items():
        if k not in args and "" not in acc:
            return f"missing {k!r}"
    return None


def check_item(item: Dict[str, Any], calls: List[Tuple[str, Dict[str, Any]]]) -> Tuple[bool, str]:
    cat = item["category"]
    if "irrelevance" in cat:
        return (not calls, "ok" if not calls else f"called {calls[0][0]!r}")
    if cat == "live_relevance":
        return (bool(calls), "ok" if calls else "no call")
    gt = item["ground_truth"]
    docs = {norm_name(f["name"]): f for f in item["function"]}
    if not calls:
        return False, "no call"
    if len(calls) != len(gt):
        return False, f"{len(calls)} calls, want {len(gt)}"
    if len(gt) == 1:
        reason = check_call(calls[0], gt[0], docs)
        return reason is None, reason or "ok"
    if len(gt) <= 6:
        for perm in permutations(range(len(gt))):
            if all(check_call(calls[i], gt[j], docs) is None for i, j in zip(range(len(calls)), perm)):
                return True, "ok"
        return False, "no matching assignment"
    remaining = list(gt)
    for c in calls:
        hit = next((g for g in remaining if check_call(c, g, docs) is None), None)
        if hit is None:
            return False, f"unmatched call {c[0]!r}"
        remaining.remove(hit)
    return True, "ok"


def _pick(params: Dict[str, List[Any]]) -> Dict[str, Any]:
    out = {}
    for k, acc in params.items():
        vals = [a for a in acc if not (isinstance(a, str) and a == "")]
        if vals:
            out[k] = _pick_value(vals[0])
    return out


def _pick_value(v: Any) -> Any:
    if isinstance(v, dict):
        return _pick({k: (a if isinstance(a, list) else [a]) for k, a in v.items()})
    if isinstance(v, list):
        return [_pick_value(x) for x in v]
    return v


def oracle_text(tokenizer, item: Dict[str, Any], tools) -> str:
    """Ground-truth calls rendered through the model's chat template."""
    tool_calls = []
    for g in item["ground_truth"]:
        (name, params), = g.items()
        tool_calls.append({"type": "function", "function": {"name": name, "arguments": _pick(params)}})
    if not tool_calls:
        return "I can't help with that using the available tools."
    msgs = list(item["question"][0]) + [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
    text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False)
    prefix = tokenizer.apply_chat_template(item["question"][0], tools=tools, tokenize=False)
    tail = text[len(prefix) :] if text.startswith(prefix) else text
    start, end = tail.find("<tool_call>"), tail.rfind("</tool_call>")
    return tail[start : end + len("</tool_call>")] if start != -1 and end != -1 else tail


def main(argv=None):
    from .generation import Generator, add_generation_args

    p = base_parser("tools", __doc__)
    p.add_argument("--categories", default=DEFAULT_CATEGORIES)
    p.add_argument("--oracle", action="store_true", help="score ground-truth calls rendered in native format")
    p.add_argument("--keep-inconsistent", action="store_true", help="do not drop INCONSISTENT_IDS")
    add_generation_args(p, thinking_budget=2048, answer_tokens=1024)
    args = p.parse_args(argv)
    started = time.time()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if args.smoke:
        categories = [c for c in categories if c in ("simple", "multiple", "parallel", "irrelevance")]
    default_limit = 2 if args.smoke else 50
    limit = args.limit if args.limit is not None else default_limit
    items = []
    for c in categories:
        rows = bfcl(c, smoke=args.smoke)
        if not args.keep_inconsistent:
            rows = [r for r in rows if r["id"] not in INCONSISTENT_IDS]
        items += take(rows, limit)

    gen = None
    if args.oracle:
        tokenizer = load_tokenizer(args.model)
    else:
        model, processor = load_model(args)
        gen = Generator.from_args(args, model, processor)
        tokenizer = gen.tokenizer
    parser_name, module = get_tool_module(tokenizer)
    settings: Dict[str, Any] = {"categories": categories, "tool_parser": parser_name, "per_category_limit": limit}
    if gen:
        settings.update(gen.settings())
    else:
        settings["oracle"] = True

    samples = []
    ck = Checkpoint(args, "tools", settings) if gen else None
    for n, item in enumerate(items):
        tools = to_openai_tools(item["function"])
        sample = ck.get(item["id"]) if ck else None
        if sample is None:
            if gen:
                out = gen.generate(item["question"][0], tools=tools)
                text, rec = out.answer, out.record()
            else:
                text, rec = oracle_text(tokenizer, item, tools), {}
            calls = parse_calls(text, module, tools)
            ok, reason = check_item(item, calls)
            sample = {
                "id": item["id"],
                "category": item["category"],
                "correct": ok,
                "reason": reason,
                "calls": [{"name": c[0], "arguments": c[1]} for c in calls],
                "output": text if not gen else None,
                **rec,
            }
            if ck:
                ck.add(item["id"], sample)
        samples.append(sample)
        print(f"[tools] {n + 1}/{len(items)} {item['id']} ok={sample['correct']} {sample['reason']}", flush=True)

    breakdown = {}
    for c in categories:
        rows = [s for s in samples if s["category"] == c]
        if rows:
            breakdown[c] = {
                "n": len(rows),
                "accuracy": mean(r["correct"] for r in rows),
                "no_call": mean(r["reason"] == "no call" for r in rows),
            }
    summary: Dict[str, Any] = {f"{c}_acc": b["accuracy"] for c, b in breakdown.items()}
    summary["accuracy"] = mean(b["accuracy"] for b in breakdown.values())
    summary["n"] = len(samples)
    summary["headline"] = "accuracy"
    write_result(args, "tools", summary=summary, breakdown=breakdown, samples=samples, settings=settings, started=started)
    if ck:
        ck.finish()


if __name__ == "__main__":
    main()
