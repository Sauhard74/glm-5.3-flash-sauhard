import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GB = 1024**3
DEFAULT_RESULTS_DIR = "results"
DEFAULT_MEMORY_LIMIT_GB = 80.0
DEFAULT_MAX_WEIGHTS_GB = 80.0

FAMILIES = {"qwen4_exp": "qwen", "glm5_next": "glm"}


def base_parser(suite: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"evals.{suite}", description=description)
    p.add_argument("--model", required=True, help="model directory (mlx_vlm.load-able)")
    p.add_argument("--name", help="results name (default: basename of --model)")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--out", help="explicit output json path (overrides results-dir/name)")
    p.add_argument("--limit", type=int, help="max items per dataset/category/task")
    p.add_argument("--smoke", action="store_true", help="tiny offline run to check the pipeline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse finished items from <suite>.partial.jsonl when settings match",
    )
    p.add_argument(
        "--memory-limit-gb",
        type=float,
        default=DEFAULT_MEMORY_LIMIT_GB,
        help="mx.set_memory_limit for this process",
    )
    p.add_argument(
        "--max-weights-gb",
        type=float,
        default=float(os.environ.get("LOCAL_AI_MAX_WEIGHTS_GB", DEFAULT_MAX_WEIGHTS_GB)),
        help="refuse to load models whose weights exceed this (env LOCAL_AI_MAX_WEIGHTS_GB)",
    )
    return p


def resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    from mlx_vlm.utils import get_model_path

    return Path(get_model_path(model))


def read_config(model_path: Path) -> Dict[str, Any]:
    with open(model_path / "config.json") as f:
        return json.load(f)


def model_family(config: Dict[str, Any]) -> str:
    return FAMILIES.get(config.get("model_type", ""), config.get("model_type", "unknown"))


def weights_gb(model_path: Path) -> float:
    return sum(p.stat().st_size for p in model_path.glob("*.safetensors")) / GB


def results_name(args) -> str:
    return args.name or Path(args.model.rstrip("/")).name


def result_path(args, suite: str) -> Path:
    if args.out:
        return Path(args.out)
    return Path(args.results_dir) / results_name(args) / f"{suite}.json"


def set_memory_limits(memory_limit_gb: float) -> None:
    import mlx.core as mx

    if memory_limit_gb and memory_limit_gb > 0:
        mx.set_memory_limit(int(memory_limit_gb * GB))


def load_model(args, *, lazy: bool = False):
    """Load (model, processor) with the co-tenant memory guards applied."""
    import mlx.core as mx
    from mlx_vlm import load

    path = resolve_model_path(args.model)
    size = weights_gb(path)
    if args.max_weights_gb and size > args.max_weights_gb:
        raise SystemExit(
            f"refusing to load {path}: {size:.1f} GB of weights > --max-weights-gb {args.max_weights_gb}"
        )
    set_memory_limits(args.memory_limit_gb)
    mx.random.seed(args.seed)
    model, processor = load(str(path), lazy=lazy)
    return model, processor


def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(resolve_model_path(model)))


def peak_memory_gb() -> float:
    import mlx.core as mx

    return mx.get_peak_memory() / GB


def versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("mlx", "mlx-lm", "mlx-vlm", "transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_result(
    args,
    suite: str,
    *,
    summary: Dict[str, Any],
    breakdown: Optional[Dict[str, Any]] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    settings: Optional[Dict[str, Any]] = None,
    started: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = result_path(args, suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    try:
        family = model_family(read_config(model_path))
    except (OSError, json.JSONDecodeError):
        family = "unknown"
    try:
        mem = peak_memory_gb()
    except Exception:
        mem = None
    record = {
        "suite": suite,
        "model": results_name(args),
        "model_path": str(model_path),
        "family": family,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "smoke": bool(getattr(args, "smoke", False)),
        "limit": getattr(args, "limit", None),
        "settings": settings or {},
        "summary": summary,
        "breakdown": breakdown or {},
        "samples": samples or [],
        "runtime": {
            "seconds": round(time.time() - started, 2) if started else None,
            "peak_memory_gb": mem,
            "versions": versions(),
        },
    }
    if extra:
        record.update(extra)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, default=_json_default)
    tmp.replace(path)
    print(f"[{suite}] wrote {path}")
    print(json.dumps(summary, indent=2, default=_json_default))
    return path


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def take(items: List[Any], limit: Optional[int]) -> List[Any]:
    return items if limit is None else items[:limit]


class Checkpoint:
    """Append-only per-item log next to the result file so long runs can resume.
    Removed once the final result json is written."""

    def __init__(self, args, suite: str, settings: Dict[str, Any]):
        self.path = result_path(args, suite).with_suffix(".partial.jsonl")
        blob = json.dumps({"suite": suite, "settings": settings, "smoke": bool(args.smoke)}, sort_keys=True, default=str)
        self.fingerprint = hashlib.sha256(blob.encode()).hexdigest()[:16]
        self.done: Dict[str, Dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if getattr(args, "resume", False) and self.path.exists():
            with open(self.path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            if lines and lines[0].get("fingerprint") == self.fingerprint:
                self.done = {r["_key"]: r["record"] for r in lines[1:]}
                print(f"[{suite}] resuming: {len(self.done)} items from {self.path}")
                return
            print(f"[{suite}] settings changed; ignoring {self.path}")
        with open(self.path, "w") as f:
            f.write(json.dumps({"fingerprint": self.fingerprint}) + "\n")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.done.get(key)

    def add(self, key: str, record: Dict[str, Any]) -> None:
        self.done[key] = record
        with open(self.path, "a") as f:
            f.write(json.dumps({"_key": key, "record": record}, default=_json_default) + "\n")

    def finish(self) -> None:
        self.path.unlink(missing_ok=True)
