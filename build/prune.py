"""REAP expert selection: uniform per-layer top-K or global ranking with per-layer min-keep."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

from .calib_io import combine_domains
from .families import MoELayer
from .source import Source

log = logging.getLogger(__name__)


@dataclass
class ExpertScores:
    score: np.ndarray  # ranking criterion (higher = keep)
    mass: np.ndarray  # expected routed output share, sums to 1 (prune-loss proxy)
    freq: Optional[np.ndarray]
    source: str  # "calib:reap", "calib:freq", "weights"


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
    s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / max(len(x), 1))


def calib_scores(
    stats, num_experts: int, metric: str = "reap", domain_weights=None
) -> Optional[ExpertScores]:
    if stats is None or not stats.has_experts:
        return None
    reap = combine_domains(stats, "reap", domain_weights)
    freq = combine_domains(stats, "freq", domain_weights)
    for name, arr in (("reap", reap), ("freq", freq)):
        if arr is not None and len(arr) != num_experts:
            raise ValueError(f"calib {name} has {len(arr)} experts, checkpoint has {num_experts}")
    if reap is None and freq is None:
        return None
    if metric == "reap" and reap is None:
        metric = "freq"
    if metric == "freq" and freq is None:
        metric = "reap"
    if reap is not None and freq is not None:
        mass = _norm(_norm(freq) * reap)
    else:
        mass = _norm(reap if reap is not None else freq)
    score = {"reap": reap, "freq": freq, "mass": mass}[metric]
    if score is None:
        score = mass
    return ExpertScores(np.asarray(score, dtype=np.float64), mass, freq, f"calib:{metric}")


def weight_scores(source: Source, layer: MoELayer, experts_down_key=None) -> ExpertScores:
    """Data-free fallback: ||router row|| * ||down_proj_e||_F (used when calib has no stats)."""
    gate_key = next((k for k in layer.router_keys if k.endswith("gate.weight")), None)
    if gate_key is None:
        raise ValueError(f"no router weight for MoE block {layer.name}")
    router = source.read(gate_key).astype(mx.float32)
    rnorm = np.array(mx.linalg.norm(router, axis=-1))
    down = np.zeros(layer.num_experts)
    if "down_proj" in layer.per_expert:
        for e, key in layer.per_expert["down_proj"].items():
            w = source.read(key).astype(mx.float32)
            down[e] = float(mx.linalg.norm(w.reshape(-1)))
            mx.clear_cache()
    else:
        key = next(k for k in layer.fused_keys if "down_proj" in k)
        for start in range(0, layer.num_experts, 32):
            rows = list(range(start, min(start + 32, layer.num_experts)))
            w = source.read(key, rows=rows).astype(mx.float32)
            down[rows] = np.array(mx.sqrt((w * w).sum(axis=(1, 2))))
            mx.clear_cache()
    score = rnorm * down
    return ExpertScores(score, _norm(score), None, "weights")


def _rank(scores: ExpertScores) -> np.ndarray:
    tie = scores.freq if scores.freq is not None else scores.mass
    return np.lexsort((-np.arange(len(scores.score)), tie, scores.score))[::-1]


def keep_top(scores: ExpertScores, k: int) -> List[int]:
    return sorted(int(i) for i in _rank(scores)[:k])


def uniform_keep(all_scores: Dict[str, ExpertScores], k: int) -> Dict[str, List[int]]:
    return {name: keep_top(s, k) for name, s in all_scores.items()}


def global_keep(
    all_scores: Dict[str, ExpertScores], total: int, min_keep: Dict[str, int]
) -> Dict[str, List[int]]:
    """Keep ``total`` experts across layers by per-layer-normalized score, each layer >= min_keep."""
    keep: Dict[str, List[int]] = {}
    pool = []
    for name, s in all_scores.items():
        order = _rank(s)
        m = min(min_keep[name], len(order))
        keep[name] = [int(i) for i in order[:m]]
        share = _norm(s.score)
        pool.extend((float(share[i]), name, int(i)) for i in order[m:])
    remaining = total - sum(len(v) for v in keep.values())
    if remaining < 0:
        raise ValueError(f"total keep {total} is below the per-layer minimum {total - remaining}")
    pool.sort(key=lambda t: (-t[0], t[1], t[2]))
    for _, name, i in pool[:remaining]:
        keep[name].append(i)
    return {name: sorted(v) for name, v in keep.items()}


def prune_loss(scores: ExpertScores, keep: List[int]) -> float:
    """Fraction of the layer's expected routed output mass removed by pruning."""
    kept = np.zeros(len(scores.mass), dtype=bool)
    kept[keep] = True
    return float(scores.mass[~kept].sum())


def parse_prune(value: str, num_experts: int, min_keep: int = 1):
    """'none'|'0' -> keep all; 0<r<1 -> prune fraction r; integer >= 1 -> keep count; 'auto'.
    A keep count below ``min_keep`` (the router's top_k) is rejected."""
    v = str(value).strip().lower()
    if v in ("none", "0", "0.0", "off", "no"):
        return ("keep", num_experts)
    if v == "auto":
        return ("auto", None)
    x = float(v)
    if 0 < x < 1:
        k = max(1, int(round(num_experts * (1 - x))))
    elif x >= 1 and x == int(x):
        k = min(int(x), num_experts)
    else:
        raise ValueError(f"bad --prune value {value!r}")
    if k < min_keep:
        raise ValueError(f"--prune {value} keeps {k} experts, below num_experts_per_tok={min_keep}")
    return ("keep", k)
