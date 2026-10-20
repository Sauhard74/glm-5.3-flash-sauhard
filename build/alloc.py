"""Per-tensor (bits, group) allocation: minimize sum of sensitivities s.t. exact bytes <= budget.

Sensitivities are NMSE (relative squared output error, as calib/ measures it), so they add as energy."""

import heapq
import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

log = logging.getLogger(__name__)

BF16 = 16  # pseudo bit-width meaning "keep unquantized"
DEFAULT_BITS = (2, 3, 4, 5, 6, 8)
DEFAULT_GROUPS = (32, 64)


def quantized_bytes(numel: int, in_dim: int, bits: int, group: int, scale_itemsize: int = 2) -> int:
    """Exact bytes of mx.quantize(affine) output: packed uint32 weight + scales + biases."""
    rows = numel // in_dim
    packed = rows * (in_dim * bits // 32) * 4
    return packed + 2 * rows * (in_dim // group) * scale_itemsize


@dataclass
class Option:
    bits: int
    group: Optional[int]
    bytes: int
    sens: float

    @property
    def label(self) -> str:
        return "bf16" if self.bits == BF16 else f"{self.bits}b/g{self.group}"


@dataclass
class Unit:
    ns: str  # "main" or "mtp"
    path: str  # module path in that model
    shape: Tuple[int, ...]
    itemsize: int  # unquantized bytes per element (final dtype)
    kind: str
    layer: Optional[str]  # MoE/decoder layer name the tensor belongs to
    expert_axis: bool = False  # leading axis indexes experts (bytes scale with keep count)
    allowed_groups: Tuple[int, ...] = DEFAULT_GROUPS
    pinned: Optional[Tuple[int, Optional[int]]] = None
    floor: int = 0
    sens: Dict[Tuple[int, int], float] = field(default_factory=dict)
    expert_sens: Dict[int, Dict[Tuple[int, int], float]] = field(default_factory=dict)
    sens_source: str = "none"
    choice: Optional[Option] = None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.ns, self.path)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def in_dim(self) -> int:
        return int(self.shape[-1])

    def with_experts(self, k: int) -> Tuple[int, ...]:
        return (k, *self.shape[1:]) if self.expert_axis else self.shape


def classify(path: str, ple_marker: Optional[str] = None) -> str:
    if ple_marker and ple_marker in path:
        return "ple_table"
    if re.search(r"\.mlp\.gate$", path) or path.endswith("shared_expert_gate"):
        return "router"
    if ".switch_mlp." in path:
        return "expert"
    if "shared_expert" in path:
        return "shared_expert"
    if path.endswith("embed_tokens"):
        return "embed"
    if path.endswith("lm_head"):
        return "lm_head"
    if path.startswith("vision_tower"):
        return "vision"
    if "hyper_connection" in path or "_hc." in path:
        return "hc"
    if ".ple." in path:
        return "ple"
    if "self_attn" in path or "linear_attn" in path:
        return "attn"
    if ".mlp." in path:
        return "dense_mlp"
    if path.startswith(("fc_", "eh_proj")):
        return "mtp_head"
    return "other"


@lru_cache(maxsize=None)
def gaussian_proxy(bits: int, group: int) -> float:
    """Weight NMSE ||W - Q(W)||^2 / ||W||^2 of affine quantization on Gaussian data (same scale as calib sens)."""
    key = mx.random.key(1234)
    w = mx.random.normal((256, 4096), key=key)
    q = mx.quantize(w, group_size=group, bits=bits)
    d = mx.dequantize(*q, group_size=group, bits=bits)
    return float(mx.square(w - d).sum() / mx.square(w).sum())


def _median_curve(grids: Sequence[Dict[Tuple[int, int], float]]) -> Dict[Tuple[int, int], float]:
    cells: Dict[Tuple[int, int], List[float]] = {}
    for g in grids:
        for k, v in g.items():
            cells.setdefault(k, []).append(v)
    return {k: float(np.median(v)) for k, v in cells.items()}


def _scale_to_proxy(grid: Dict[Tuple[int, int], float]) -> Optional[float]:
    ratios = [v / gaussian_proxy(*k) for k, v in grid.items() if gaussian_proxy(*k) > 0 and v > 0]
    return float(np.median(ratios)) if ratios else None


def aggregate_expert_sens(unit: Unit, keep: Optional[Sequence[int]], freq: Optional[np.ndarray]):
    if not unit.expert_sens:
        return unit.sens
    ids = [e for e in (keep if keep is not None else unit.expert_sens) if e in unit.expert_sens]
    if not ids:
        return unit.sens
    w = np.array([freq[e] if freq is not None else 1.0 for e in ids], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    cells = set.intersection(*(set(unit.expert_sens[e]) for e in ids))
    return {c: float(sum(wi * unit.expert_sens[e][c] for wi, e in zip(w, ids)) / w.sum()) for c in cells}


def fill_sensitivities(units: List[Unit], bits=DEFAULT_BITS, groups=DEFAULT_GROUPS):
    """Give every free unit a full (bits, group) curve: measured > same-kind median > scaled proxy."""
    for u in units:
        if u.expert_sens and not u.sens:
            u.sens = dict(aggregate_expert_sens(u, None, None))
    measured = [u for u in units if u.sens]
    by_kind: Dict[str, List[Dict]] = {}
    for u in measured:
        by_kind.setdefault(u.kind, []).append(u.sens)
    kind_curves = {k: _median_curve(v) for k, v in by_kind.items()}
    scales = [s for s in (_scale_to_proxy(u.sens) for u in measured) if s]
    global_scale = float(np.median(scales)) if scales else 1.0
    cells = [(b, g) for b in bits for g in groups]
    for u in units:
        if u.sens:
            own = _scale_to_proxy(u.sens) or global_scale
            u.sens_source = u.sens_source if u.sens_source != "none" else "calib"
            for c in cells:
                u.sens.setdefault(c, own * gaussian_proxy(*c))
            continue
        curve = kind_curves.get(u.kind)
        if curve:
            scale = _scale_to_proxy(curve) or global_scale
            u.sens = {c: curve.get(c, scale * gaussian_proxy(*c)) for c in cells}
            u.sens_source = f"median:{u.kind}"
        else:
            u.sens = {c: global_scale * gaussian_proxy(*c) for c in cells}
            u.sens_source = "proxy"


def unit_options(
    unit: Unit,
    shape: Tuple[int, ...],
    sens: Dict[Tuple[int, int], float],
    bits=DEFAULT_BITS,
    groups=DEFAULT_GROUPS,
    allow_bf16: bool = True,
    weight: float = 1.0,
) -> List[Option]:
    numel = int(np.prod(shape))
    in_dim = int(shape[-1])
    if unit.pinned is not None:
        b, g = unit.pinned
        if b == BF16:
            return [Option(BF16, None, numel * unit.itemsize, 0.0)]
        if g is None or in_dim % g:
            g = max(x for x in unit.allowed_groups if in_dim % x == 0)
        return [Option(b, g, quantized_bytes(numel, in_dim, b, g, unit.itemsize), weight * sens.get((b, g), 0.0))]
    opts = []
    for b in bits:
        if b < unit.floor:
            continue
        for g in groups:
            if g not in unit.allowed_groups or in_dim % g or (b, g) not in sens:
                continue
            opts.append(Option(b, g, quantized_bytes(numel, in_dim, b, g, unit.itemsize), weight * sens[(b, g)]))
    if allow_bf16 or not opts:
        opts.append(Option(BF16, None, numel * unit.itemsize, 0.0))
    return opts


@dataclass
class Solution:
    choices: List[Option]
    bytes: int
    objective: float
    method: str


def _hull(opts: List[Option]) -> List[Option]:
    """Lower-left convex hull in (bytes, sens), sorted by bytes."""
    pts = sorted(opts, key=lambda o: (o.bytes, o.sens))
    frontier = []
    for o in pts:
        if frontier and o.sens >= frontier[-1].sens:
            continue
        frontier.append(o)
    hull: List[Option] = []
    for o in frontier:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (b.sens - a.sens) * (o.bytes - a.bytes) >= (o.sens - a.sens) * (b.bytes - a.bytes):
                hull.pop()
            else:
                break
        hull.append(o)
    return hull


def solve_greedy(options: List[List[Option]], budget: int) -> Optional[Solution]:
    hulls = [_hull(o) for o in options]
    idx = [0] * len(hulls)
    used = sum(h[0].bytes for h in hulls)
    if used > budget:
        return None
    heap = []

    def push(i):
        h = hulls[i]
        j = idx[i]
        if j + 1 < len(h):
            db = h[j + 1].bytes - h[j].bytes
            ds = h[j].sens - h[j + 1].sens
            heapq.heappush(heap, (-(ds / max(db, 1)), i))

    for i in range(len(hulls)):
        push(i)
    while heap:
        _, i = heapq.heappop(heap)
        h = hulls[i]
        j = idx[i]
        db = h[j + 1].bytes - h[j].bytes
        if used + db <= budget:
            used += db
            idx[i] = j + 1
            push(i)
    choices = [h[j] for h, j in zip(hulls, idx)]
    choices, used = _fill(options, choices, used, budget)
    return Solution(choices, used, sum(c.sens for c in choices), "greedy")


def _fill(options, choices, used, budget):
    """Spend leftover bytes on any non-hull upgrade with the best sens gain per byte."""
    while True:
        best = None
        for i, opts in enumerate(options):
            cur = choices[i]
            for o in opts:
                db = o.bytes - cur.bytes
                ds = cur.sens - o.sens
                if ds > 0 and used + db <= budget:
                    r = ds / max(db, 1) if db > 0 else math.inf
                    if best is None or r > best[0]:
                        best = (r, i, o, db)
        if best is None:
            return choices, used
        _, i, o, db = best
        choices = list(choices)
        choices[i] = o
        used += db


def solve_dp(options: List[List[Option]], budget: int, buckets: int = 1 << 17) -> Optional[Solution]:
    base = [min(o, key=lambda x: (x.bytes, x.sens)) for o in options]
    base_bytes = sum(b.bytes for b in base)
    slack = budget - base_bytes
    if slack < 0:
        return None
    free = [i for i, o in enumerate(options) if len(o) > 1]
    choices = list(base)
    if free and slack > 0:
        nb = int(min(buckets, slack))
        q = slack / nb
        dp = np.zeros(nb + 1)
        arg = np.zeros((len(free), nb + 1), dtype=np.uint8)
        for r, i in enumerate(free):
            opts = options[i]
            best = np.full(nb + 1, np.inf)
            best_arg = np.zeros(nb + 1, dtype=np.uint8)
            for j, o in enumerate(opts):
                c = int(math.ceil((o.bytes - base[i].bytes) / q))
                if c > nb:
                    continue
                cand = np.full(nb + 1, np.inf)
                cand[c:] = dp[: nb + 1 - c] + o.sens
                better = cand < best
                best[better] = cand[better]
                best_arg[better] = j
            dp = best
            arg[r] = best_arg
        cap = nb
        for r in range(len(free) - 1, -1, -1):
            i = free[r]
            o = options[i][arg[r, cap]]
            choices[i] = o
            cap -= int(math.ceil((o.bytes - base[i].bytes) / q))
    used = sum(c.bytes for c in choices)
    choices, used = _fill(options, choices, used, budget)
    return Solution(choices, used, sum(c.sens for c in choices), "dp")


def solve(options: List[List[Option]], budget: int, method: str = "dp") -> Optional[Solution]:
    greedy = solve_greedy(options, budget)
    if method == "greedy" or greedy is None:
        return greedy
    dp = solve_dp(options, budget)
    if dp is None or dp.bytes > budget or greedy.objective <= dp.objective:
        return greedy
    return dp
