"""HumanEval+ / MBPP+ pass@1 with sandboxed execution.

uv run python -m evals.code --model models/<name> [--datasets humaneval+,mbpp+] [--limit N]
"""

import re
import time
from typing import Any, Dict, List

from .common import Checkpoint, base_parser, load_model, mean, take, write_result
from .data import humaneval_plus, mbpp_plus
from .parsing import extract_code
from .sandbox import run_many

HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Reply with the complete implementation "
    "(signature, docstring optional, and any imports it needs) in a single ```python code block.\n\n"
    "```python\n{prompt}```"
)
MBPP_INSTRUCTION = (
    "{prompt}\nYour code should pass this test:\n\n```python\n{test}\n```\n\n"
    "Reply with the complete implementation in a single ```python code block."
)


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    if item["dataset"] == "humaneval+":
        content = HUMANEVAL_INSTRUCTION.format(prompt=item["prompt"])
    else:
        content = MBPP_INSTRUCTION.format(prompt=item["prompt"], test=item["test_list"][0])
    return [{"role": "user", "content": content}]


def _prompt_imports(prompt: str) -> str:
    return "\n".join(
        line for line in prompt.splitlines() if re.match(r"^(from\s+\S+\s+)?import\s", line)
    )


def assemble(item: Dict[str, Any], code: str) -> Dict[str, str]:
    """Return {test_name: program} for a candidate solution."""
    if item["dataset"] == "humaneval+":
        ep = item["entry_point"]
        if not re.search(rf"def\s+{re.escape(ep)}\s*\(", code):
            code = item["prompt"] + code
        body = _prompt_imports(item["prompt"]) + "\n" + code
        return {"plus": f"{body}\n\n{item['test']}\n\ncheck({ep})\n"}
    imports = "\n".join(item["test_imports"])
    body = f"{imports}\n{code}\n"
    return {
        "base": body + "\n\n" + "\n".join(item["test_list"]) + "\n",
        "plus": body + "\n\n" + item["test"] + "\n",
    }


def oracle_completion(item: Dict[str, Any]) -> str:
    sol = item["canonical_solution"]
    if item["dataset"] == "humaneval+":
        sol = item["prompt"] + sol
    return f"```python\n{sol}\n```"


def score(items, answers, timeout: float, workers: int) -> List[Dict[str, Any]]:
    programs, index = [], []
    for i, (item, answer) in enumerate(zip(items, answers)):
        for name, prog in assemble(item, extract_code(answer)).items():
            programs.append(prog)
            index.append((i, name))
    results = run_many(programs, timeout=timeout, workers=workers)
    per_item: List[Dict[str, Any]] = [{} for _ in items]
    for (i, name), res in zip(index, results):
        per_item[i][name] = res.status
        if res.status != "pass":
            per_item[i][f"{name}_output"] = res.output[-500:]
    return per_item


def main(argv=None):
    from .generation import Generator, add_generation_args

    p = base_parser("code", __doc__)
    p.add_argument("--datasets", default="humaneval+,mbpp+")
    p.add_argument("--timeout", type=float, default=20.0, help="seconds per test program")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--oracle", action="store_true", help="score canonical solutions (harness check, no model)")
    add_generation_args(p, thinking_budget=16384, answer_tokens=4096)
    args = p.parse_args(argv)
    started = time.time()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    limit = 2 if args.smoke and args.limit is None else args.limit
    items: List[Dict[str, Any]] = []
    for d in datasets:
        loader = {"humaneval+": humaneval_plus, "mbpp+": mbpp_plus}[d]
        items += take(loader(smoke=args.smoke), limit)

    samples: List[Dict[str, Any]] = []
    answers: List[str] = []
    settings: Dict[str, Any] = {"datasets": datasets, "timeout": args.timeout}
    ck = None
    if args.oracle:
        answers = [oracle_completion(it) for it in items]
        settings["oracle"] = True
        samples = [{"id": it["id"], "dataset": it["dataset"]} for it in items]
    else:
        model, processor = load_model(args)
        gen = Generator.from_args(args, model, processor)
        settings.update(gen.settings())
        ck = Checkpoint(args, "code", settings)
        for n, item in enumerate(items):
            rec = ck.get(item["id"])
            if rec is None:
                out = gen.generate(build_messages(item))
                rec = {"id": item["id"], "dataset": item["dataset"], **out.record()}
                ck.add(item["id"], rec)
            answers.append(rec["answer"])
            samples.append(rec)
            print(f"[code] {n + 1}/{len(items)} {item['id']} tokens={rec['generation_tokens']}", flush=True)

    scores = score(items, answers, args.timeout, args.workers)
    for rec, sc, ans in zip(samples, scores, answers):
        rec.update(sc)
        rec["code"] = extract_code(ans)

    breakdown: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {}
    for d in datasets:
        rows = [s for s in samples if s["dataset"] == d]
        if not rows:
            continue
        b = {"n": len(rows), "pass@1_plus": mean(r["plus"] == "pass" for r in rows)}
        if d == "mbpp+":
            b["pass@1_base"] = mean(r["base"] == "pass" for r in rows)
        if not args.oracle:
            b["mean_gen_tokens"] = mean(r["generation_tokens"] for r in rows)
            b["thinking_truncated"] = mean(not r["thinking_closed"] for r in rows)
        breakdown[d] = b
        summary[f"{d}_pass@1"] = b["pass@1_plus"]
    summary["pass@1"] = mean(s["plus"] == "pass" for s in samples)
    summary["n"] = len(samples)
    summary["headline"] = "pass@1"
    write_result(args, "code", summary=summary, breakdown=breakdown, samples=samples, settings=settings, started=started)
    if ck:
        ck.finish()


if __name__ == "__main__":
    main()
