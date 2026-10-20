"""Tolerant loader for calib/ outputs (see docs/contracts/build.md for accepted encodings).

Everything is normalized to::

    CalibStats.layers[name] -> LayerStats   # name: "<decoder idx>", "mtp" or "global"
    LayerStats.freq / reap                  # np.ndarray[E] or None (overall)
    LayerStats.domain_freq / domain_reap    # {domain: np.ndarray[E]}
    LayerStats.sens                         # {raw tensor name: {(bits, group): float}}

Per-expert grids (npz ``sens.mlp.switch_mlp.<p>`` of shape [E, bits, groups]) become
``mlp.experts.<e>.<p>`` entries. When several keys alias one metric (``expert_freq`` and the raw
``expert_count``), the earlier key in ``_FREQ_KEYS`` / ``_REAP_KEYS`` wins.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

Grid = Dict[Tuple[int, int], float]

_FREQ_KEYS = ("expert_freq", "expert_frequency", "freq", "expert_counts", "expert_count")
_REAP_KEYS = ("expert_reap", "reap", "expert_saliency", "reap_score", "expert_reap_score")
_OVERALL = ("all", "overall", "total", "mean", "_all", "global")
_SENS_KEYS = ("sensitivities", "sensitivity", "quant_sens", "sens")
_BITS_AXES = ("grid_bits", "sens_bits", "bits")
_GROUP_AXES = ("grid_groups", "sens_groups", "sens_group_sizes", "groups", "group_sizes")
_DOMAIN_CONTAINERS = ("domains", "per_domain", "by_domain")
_BG_RE = re.compile(r"(\d+)\D+(\d+)")
_STRICT_BG_RE = re.compile(r"^[\(\[]?[a-z]{0,4}\s*(\d+)\s*[a-z_,\s/x:\-]{1,6}(\d+)\s*[\)\]]?$", re.I)


@dataclass
class LayerStats:
    freq: Optional[np.ndarray] = None
    reap: Optional[np.ndarray] = None
    domain_freq: Dict[str, np.ndarray] = field(default_factory=dict)
    domain_reap: Dict[str, np.ndarray] = field(default_factory=dict)
    sens: Dict[str, Grid] = field(default_factory=dict)
    unparsed: List[str] = field(default_factory=list)
    prio: Dict[Tuple[str, Optional[str]], int] = field(default_factory=dict, repr=False)

    def put(self, metric: str, domain: Optional[str], value, prio: int = 0):
        arr = _array(value)
        if arr is None or self.prio.get((metric, domain), prio + 1) <= prio:
            return
        self.prio[(metric, domain)] = prio
        if domain is None:
            setattr(self, metric, arr)
        else:
            getattr(self, f"domain_{metric}")[domain] = arr

    def merge(self, other: "LayerStats"):
        for (metric, domain), p in other.prio.items():
            arr = getattr(other, metric) if domain is None else getattr(other, f"domain_{metric}")[domain]
            self.put(metric, domain, arr, p)
        for k, v in other.sens.items():
            self.sens.setdefault(k, {}).update(v)
        self.unparsed.extend(other.unparsed)

    @property
    def has_experts(self) -> bool:
        return any(
            x is not None and len(x)
            for x in (self.freq, self.reap, *self.domain_freq.values(), *self.domain_reap.values())
        )


def parse_bits_group(key) -> Optional[Tuple[int, int]]:
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return int(key[0]), int(key[1])
    m = _BG_RE.search(str(key))
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_grid(obj) -> Grid:
    """Accepts {"4,64": v}, {"(4, 64)": v}, {"b4g64": v}, {4: {64: v}},
    [{"bits":4,"group_size":64,"value":v}], [[4, 64, v]]."""
    grid: Grid = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                if str(k).isdigit():
                    for g, val in v.items():
                        if _is_number(val) and str(g).isdigit():
                            grid[(int(k), int(g))] = float(val)
                continue
            bg = parse_bits_group(k)
            if bg is not None and _is_number(v):
                grid[bg] = float(v)
    elif isinstance(obj, (list, tuple)):
        for rec in obj:
            if isinstance(rec, dict):
                bits = rec.get("bits")
                group = rec.get("group", rec.get("group_size"))
                val = next(
                    (rec[k] for k in ("value", "sens", "err", "rel_err", "error", "loss") if k in rec),
                    None,
                )
                if bits is not None and group is not None and _is_number(val):
                    grid[(int(bits), int(group))] = float(val)
            elif isinstance(rec, (list, tuple)) and len(rec) == 3:
                grid[(int(rec[0]), int(rec[1]))] = float(rec[2])
    return {k: v for k, v in grid.items() if np.isfinite(v)}


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)


def _array(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return arr if arr.size else None


def _layer_name(path: Path) -> str:
    stem = path.stem.lower()
    if "mtp" in stem:
        return "mtp"
    m = re.search(r"layer[_\-]?(\d+)", stem)
    if m:
        return str(int(m.group(1)))
    return "global"


def _split_metric(key: str, metrics) -> Optional[Tuple[str, Optional[str], int]]:
    """'expert_reap' -> ('reap', None, 0); 'expert_reap/code' or 'expert_reap__code' -> ('reap', 'code', 0).
    The last item is the alias rank (lower = preferred)."""
    for metric_keys, metric in metrics:
        if key in metric_keys:
            return metric, None, metric_keys.index(key)
    best = None
    for metric_keys, metric in metrics:
        for mk in metric_keys:
            if key.startswith(mk) and len(key) > len(mk) and key[len(mk)] in "/:._|-":
                if best is None or len(mk) > len(best[0]):
                    best = (mk, metric, metric_keys.index(mk))
    if best is None:
        return None
    domain = key[len(best[0]) :].lstrip("/:._|-")
    return best[1], (None if domain in _OVERALL else domain), best[2]


_METRICS = ((_FREQ_KEYS, "freq"), (_REAP_KEYS, "reap"))


def _has_values(grid) -> bool:
    if isinstance(grid, dict):
        return any(_has_values(v) for v in grid.values())
    if isinstance(grid, (list, tuple)):
        return any(_has_values(v) for v in grid)
    return _is_number(grid) and np.isfinite(grid)


def _parse_tensors(stats: LayerStats, tensors: dict):
    """calib/'s ``"tensors": {name: {"kind", "shape", "sens": {"4_64": v}, ...}}``."""
    for name, entry in tensors.items():
        if not isinstance(entry, dict):
            continue
        grid = next((entry[k] for k in _SENS_KEYS if k in entry), None)
        parsed = parse_grid(grid) if grid is not None else {}
        if parsed:
            stats.sens.setdefault(str(name), {}).update(parsed)
        elif _has_values(grid):
            stats.unparsed.append(str(name))


def _parse_json(obj: dict) -> LayerStats:
    stats = LayerStats()
    for key, value in obj.items():
        if key == "tensors" and isinstance(value, dict):
            _parse_tensors(stats, value)
            continue
        if key == "moe" and isinstance(value, dict):
            stats.merge(_parse_json(value))
            continue
        if key in _SENS_KEYS and isinstance(value, dict):
            if value and all(
                isinstance(v, dict) and _STRICT_BG_RE.match(str(k)) and not any(str(n).isdigit() for n in v)
                for k, v in value.items()
            ):
                inverted: Dict[str, dict] = {}
                for bg, per_name in value.items():
                    for name, v in per_name.items():
                        inverted.setdefault(name, {})[bg] = v
                value = inverted
            for name, grid in value.items():
                parsed = parse_grid(grid)
                if parsed:
                    stats.sens.setdefault(str(name), {}).update(parsed)
            continue
        if key in _DOMAIN_CONTAINERS and isinstance(value, dict):
            for domain, sub in value.items():
                if not isinstance(sub, dict):
                    continue
                for sk, sv in sub.items():
                    hit = _split_metric(sk, _METRICS)
                    if hit:
                        stats.put(hit[0], None if domain in _OVERALL else domain, sv, hit[2])
            continue
        hit = _split_metric(key, _METRICS)
        if hit is None:
            continue
        metric, domain, prio = hit
        if isinstance(value, dict):
            for d, sv in value.items():
                stats.put(metric, None if d in _OVERALL else d, sv, prio)
        else:
            stats.put(metric, domain, value, prio)
    return stats


def _expert_name(name: str, e: int) -> str:
    base, _, leaf = name.rpartition(".")
    if base == "switch_mlp" or base.endswith(".switch_mlp"):
        base = base[: -len("switch_mlp")].rstrip(".")
    return f"{base}.experts.{e}.{leaf}" if base else f"experts.{e}.{leaf}"


def _array_grid(arr: np.ndarray, bits_axis, group_axis) -> Grid:
    return {
        (int(b), int(g)): float(arr[i, j])
        for i, b in enumerate(bits_axis)
        for j, g in enumerate(group_axis)
        if np.isfinite(arr[i, j])
    }


def _parse_npz(data) -> LayerStats:
    stats = LayerStats()
    keys = list(data.keys())
    bits_axis = next((np.asarray(data[k]).reshape(-1) for k in _BITS_AXES if k in data), None)
    group_axis = next((np.asarray(data[k]).reshape(-1) for k in _GROUP_AXES if k in data), None)
    for key in keys:
        if key in _BITS_AXES or key in _GROUP_AXES:
            continue
        value = data[key]
        sens_prefix = next(
            (p for p in _SENS_KEYS if key.startswith(p) and len(key) > len(p) and key[len(p)] in "/:|_."), None
        )
        if sens_prefix is not None:
            rest = key[len(sens_prefix) :].lstrip("/:|_.")
            parts = [p for p in re.split(r"[/:|]|__", rest) if p]
            bg = parse_bits_group(parts[-1]) if len(parts) > 1 else None
            shape = (len(bits_axis), len(group_axis)) if bits_axis is not None and group_axis is not None else None
            arr = np.asarray(value, dtype=np.float64)
            if bg is not None and arr.size == 1:
                name = "/".join(parts[:-1])
                stats.sens.setdefault(name, {})[bg] = float(arr.reshape(-1)[0])
            elif shape is not None and arr.shape == shape:
                grid = _array_grid(arr, bits_axis, group_axis)
                if grid:
                    stats.sens.setdefault("/".join(parts), {}).update(grid)
            elif shape is not None and arr.ndim == 3 and arr.shape[1:] == shape:
                name = "/".join(parts)
                for e in range(arr.shape[0]):
                    grid = _array_grid(arr[e], bits_axis, group_axis)
                    if grid:
                        stats.sens.setdefault(_expert_name(name, e), {}).update(grid)
            elif np.isfinite(arr).any():
                stats.unparsed.append(key)
            continue
        hit = _split_metric(key, _METRICS)
        if hit:
            stats.put(*hit[:2], value, hit[2])
    return stats


class CalibStats:
    def __init__(self, layers: Optional[Dict[str, LayerStats]] = None, path=None):
        self.layers: Dict[str, LayerStats] = layers or {}
        self.path = path

    @classmethod
    def load(cls, path) -> "CalibStats":
        path = Path(path) if path else None
        layers: Dict[str, LayerStats] = {}
        if path is None or not path.exists():
            if path is not None:
                log.warning("calib dir %s does not exist; using data-free fallbacks", path)
            return cls(layers, path)
        files = sorted(list(path.glob("*.json")) + list(path.glob("*.npz")))
        for file in files:
            try:
                if file.suffix == ".json":
                    with open(file) as f:
                        obj = json.load(f)
                    if not isinstance(obj, dict):
                        continue
                    stats = _parse_json(obj)
                    name = _layer_name(file)
                    if name == "global" and isinstance(obj.get("layer"), int):
                        name = str(obj["layer"])
                else:
                    with np.load(file, allow_pickle=False) as data:
                        stats = _parse_npz(data)
                    name = _layer_name(file)
            except Exception as e:  # tolerate unrelated/partial files
                log.warning("skipping unreadable calib file %s: %s", file, e)
                continue
            layers.setdefault(name, LayerStats()).merge(stats)
        unparsed = [f"{n}:{k}" for n, st in layers.items() for k in st.unparsed]
        if unparsed:
            log.warning("%d calib sensitivity entries could not be parsed (e.g. %s)", len(unparsed), unparsed[:5])
        return cls(layers, path)

    def get(self, name: str, aliases=()) -> Optional[LayerStats]:
        for n in (name, *aliases):
            if n in self.layers:
                return self.layers[n]
        return None

    def sens_items(self):
        for layer, stats in self.layers.items():
            for name, grid in stats.sens.items():
                yield layer, name, grid


def combine_domains(
    stats: LayerStats, metric: str, domain_weights: Optional[Dict[str, float]] = None
) -> Optional[np.ndarray]:
    """Overall metric, or a weighted mix of per-domain metrics when weights are given.
    Per-domain frequencies are normalized first, so raw counts are mixed domain-balanced."""
    overall = getattr(stats, metric)
    per_domain = getattr(stats, f"domain_{metric}")
    if domain_weights:
        usable = {d: w for d, w in domain_weights.items() if d in per_domain and w > 0}
        if usable:
            num = 0.0
            den = 0.0
            for d, w in usable.items():
                arr = per_domain[d]
                if metric == "reap" and d in stats.domain_freq:
                    f = _normalize(stats.domain_freq[d])
                    num = num + w * f * arr
                    den = den + w * f
                else:
                    num = num + w * (_normalize(arr) if metric == "freq" else arr)
                    den = den + w
            if metric == "reap":
                return np.where(np.asarray(den) > 0, num / np.maximum(den, 1e-30), 0.0)
            return num / den
    if overall is not None:
        return overall
    if per_domain:
        arrs = list(per_domain.values())
        if metric == "reap" and all(d in stats.domain_freq for d in per_domain):
            fs = [_normalize(stats.domain_freq[d]) for d in per_domain]
            den = sum(fs)
            return np.where(den > 0, sum(f * a for f, a in zip(fs, arrs)) / np.maximum(den, 1e-30), 0.0)
        if metric == "freq":
            arrs = [_normalize(a) for a in arrs]
        return sum(arrs) / len(arrs)
    return None


def _normalize(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    return x / s if s > 0 else x
