"""AIME 2024/2025 + MATH-500 subset accuracy (answers extracted after </think>).

uv run python -m evals.math --model models/<name> [--datasets aime24,aime25,math500] [--samples 4]
"""

import time
from typing import Any, Dict, List

from .common import Checkpoint, base_parser, load_model, mean, take, write_result
from .data import math_problems
from .parsing import extract_integer, extract_math_answer, math_equal

INSTRUCTION = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": INSTRUCTION.format(problem=item["problem"])}]


def grade(item: Dict[str, Any], answer: str, reasoning: str = "") -> Dict[str, Any]:
    """Extract from the final answer; fall back to the reasoning tail if the
    answer section is empty (thinking ran out of budget)."""
    source = "answer"
    pred = extract_math_answer(answer) if answer else None
    if pred is None and reasoning:
        pred = extract_math_answer(reasoning[-4000:])
        source = "reasoning" if pred is not None else "none"
    elif pred is None:
        source = "none"
    if item.get("integer"):
        value = extract_integer(pred)
        try:
            correct = value is not None and value == int(str(item["answer"]).strip())
        except ValueError:
            correct = math_equal(pred, item["answer"])
    else:
        correct = math_equal(pred, item["answer"])
    return {"pred": pred, "correct": bool(correct), "extracted_from": source}


def main(argv=None):
    from .generation import Generator, add_generation_args

    p = base_parser("math", __doc__)
    p.add_argument("--datasets", default="aime24,aime25,math500")
    p.add_argument("--math500-n", type=int, default=100, help="evenly strided MATH-500 subset size")
    p.add_argument("--samples", type=int, default=1, help="generations per problem (mean@k)")
    p.add_argument("--oracle", action="store_true", help="grade the reference answers (harness check)")
    add_generation_args(p, thinking_budget=24576, answer_tokens=2048)
    args = p.parse_args(argv)
    started = time.time()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_items = math_problems(datasets, smoke=args.smoke, math500_n=args.math500_n)
    limit = 2 if args.smoke and args.limit is None else args.limit
    items = []
    for d in datasets:
        items += take([it for it in all_items if it["dataset"] == d], limit)

    settings: Dict[str, Any] = {"datasets": datasets, "samples": args.samples, "math500_n": args.math500_n}
    samples: List[Dict[str, Any]] = []
    ck = None
    if args.oracle:
        settings["oracle"] = True
        for item in items:
            g = grade(item, f"The answer is $\\boxed{{{item['answer']}}}$.")
            samples.append({"id": item["id"], "dataset": item["dataset"], "sample": 0, "answer_gt": item["answer"], **g})
    else:
        model, processor = load_model(args)
        gen = Generator.from_args(args, model, processor)
        settings.update(gen.settings())
        ck = Checkpoint(args, "math", settings)
        for n, item in enumerate(items):
            for k in range(args.samples):
                key = f"{item['id']}#{k}"
                rec = ck.get(key)
                if rec is None:
                    out = gen.generate(build_messages(item))
                    g = grade(item, out.answer, out.reasoning)
                    rec = {
                        "id": item["id"],
                        "dataset": item["dataset"],
                        "sample": k,
                        "answer_gt": item["answer"],
                        **g,
                        **out.record(),
                    }
                    ck.add(key, rec)
                samples.append(rec)
                print(
                    f"[math] {n + 1}/{len(items)} {item['id']} s{k} pred={rec['pred']!r} "
                    f"gt={item['answer']!r} ok={rec['correct']} tokens={rec['generation_tokens']}",
                    flush=True,
                )

    breakdown: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {}
    for d in datasets:
        rows = [s for s in samples if s["dataset"] == d]
        if not rows:
            continue
        b: Dict[str, Any] = {"n_problems": len({r["id"] for r in rows}), "accuracy": mean(r["correct"] for r in rows)}
        if not args.oracle:
            b["mean_gen_tokens"] = mean(r["generation_tokens"] for r in rows)
            b["thinking_truncated"] = mean(not r["thinking_closed"] for r in rows)
            b["no_answer"] = mean(r["pred"] is None for r in rows)
        breakdown[d] = b
        summary[f"{d}_acc"] = b["accuracy"]
    summary["accuracy"] = mean(b["accuracy"] for b in breakdown.values())
    summary["n"] = len(samples)
    summary["headline"] = "accuracy"
    write_result(args, "math", summary=summary, breakdown=breakdown, samples=samples, settings=settings, started=started)
    if ck:
        ck.finish()


if __name__ == "__main__":
    main()
