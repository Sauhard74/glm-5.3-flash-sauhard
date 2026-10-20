"""Benchmark loaders. Real data comes from the HF hub (dataset repos, cached by
huggingface_hub); ``smoke=True`` uses the small offline fixtures in evals/fixtures
which follow the same raw schema, so both paths share the normalizers."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

FIXTURES = Path(__file__).parent / "fixtures"

HUMANEVAL_PLUS = ("evalplus/humanevalplus", "test.jsonl")
MBPP_PLUS = ("evalplus/mbppplus", "data/test-00000-of-00001-d5781c9c51e02795.parquet")
AIME24 = ("HuggingFaceH4/aime_2024", "data/train-00000-of-00001.parquet")
AIME25 = ("MathArena/aime_2025", "data/train-00000-of-00001.parquet")
MATH500 = ("HuggingFaceH4/MATH-500", "test.jsonl")
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_NO_ANSWER = {"irrelevance", "live_irrelevance", "live_relevance"}


def hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_table(path: Path) -> List[Dict[str, Any]]:
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return read_jsonl(path)


def fixture(name: str) -> List[Dict[str, Any]]:
    return read_jsonl(FIXTURES / f"{name}.jsonl")


def humaneval_plus(smoke: bool = False) -> List[Dict[str, Any]]:
    rows = fixture("humaneval") if smoke else read_table(hf_file(*HUMANEVAL_PLUS))
    return [
        {
            "id": r["task_id"],
            "dataset": "humaneval+",
            "prompt": r["prompt"],
            "entry_point": r["entry_point"],
            "test": r["test"],
            "canonical_solution": r.get("canonical_solution", ""),
        }
        for r in rows
    ]


def mbpp_plus(smoke: bool = False) -> List[Dict[str, Any]]:
    rows = fixture("mbpp") if smoke else read_table(hf_file(*MBPP_PLUS))
    return [
        {
            "id": f"Mbpp/{r['task_id']}",
            "dataset": "mbpp+",
            "prompt": r["prompt"],
            "test_list": list(r["test_list"]),
            "test_imports": list(r.get("test_imports") or []),
            "test": r["test"],
            "canonical_solution": r.get("code", ""),
        }
        for r in rows
    ]


def _aime(rows, dataset: str) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        rid = r.get("id", r.get("problem_idx", i))
        out.append(
            {
                "id": f"{dataset}/{rid}",
                "dataset": dataset,
                "problem": r["problem"],
                "answer": str(r["answer"]).strip(),
                "integer": True,
            }
        )
    return out


def math_problems(datasets: List[str], smoke: bool = False, math500_n: Optional[int] = None):
    if smoke:
        rows = fixture("math")
        return [dict(r, integer=r["dataset"].startswith("aime")) for r in rows if r["dataset"] in datasets]
    out: List[Dict[str, Any]] = []
    for name in datasets:
        if name == "aime24":
            out += _aime(read_table(hf_file(*AIME24)), "aime24")
        elif name == "aime25":
            out += _aime(read_table(hf_file(*AIME25)), "aime25")
        elif name == "math500":
            rows = read_table(hf_file(*MATH500))
            if math500_n and math500_n < len(rows):
                stride = len(rows) / math500_n
                rows = [rows[int(i * stride)] for i in range(math500_n)]
            out += [
                {
                    "id": r["unique_id"],
                    "dataset": "math500",
                    "problem": r["problem"],
                    "answer": r["answer"],
                    "integer": False,
                    "subject": r.get("subject"),
                    "level": r.get("level"),
                }
                for r in rows
            ]
        else:
            raise ValueError(f"unknown math dataset {name!r}")
    return out


def bfcl(category: str, smoke: bool = False) -> List[Dict[str, Any]]:
    if smoke:
        return [r for r in fixture("bfcl") if r["category"] == category]
    rows = read_jsonl(hf_file(BFCL_REPO, f"BFCL_v3_{category}.json"))
    answers: Dict[str, Any] = {}
    if category not in BFCL_NO_ANSWER:
        for a in read_jsonl(hf_file(BFCL_REPO, f"possible_answer/BFCL_v3_{category}.json")):
            answers[a["id"]] = a["ground_truth"]
    out = []
    for r in rows:
        if category not in BFCL_NO_ANSWER and r["id"] not in answers:
            continue
        out.append(
            {
                "id": r["id"],
                "category": category,
                "question": r["question"],
                "function": r["function"],
                "ground_truth": answers.get(r["id"], []),
            }
        )
    return out
