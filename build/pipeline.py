"""Planning + streaming build of an mlx-vlm model directory from a HF snapshot."""

import copy
import importlib
import json
import logging
import re
import shutil
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_vlm.utils import skip_multimodal_module, update_module_configs

from . import alloc, prune, round as rounding, shim
from .calib_io import CalibStats
from .families import Family, MoELayer, family_for, find_moe_layers
from .source import Source

log = logging.getLogger(__name__)

GB = 1_000_000_000
TOKENIZER_GLOBS = ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken")
SKIP_COPY = {
    "config.json",
    "model.safetensors.index.json",
    "hf_quant_config.json",
    "quantization_config.json",
    "build_summary.json",
}
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
ROUTER_KEEP_DTYPE = re.compile(r"\.mlp\.gate\.(weight|e_score_correction_bias)$")


@dataclass
class BuildOptions:
    model: Optional[str]
    src: str
    out: str
    calib: Optional[str] = None
    prune: str = "none"
    prune_mode: str = "uniform"  # uniform | global
    budget_gb: float = 68.0
    drop_vision: bool = False
    mtp: str = "prune"  # prune | intact | drop
    bits: Tuple[int, ...] = alloc.DEFAULT_BITS
    groups: Tuple[int, ...] = alloc.DEFAULT_GROUPS
    allow_bf16: bool = True
    solver: str = "dp"
    pins: List[Tuple[str, int, Optional[int]]] = field(default_factory=list)
    floors: List[Tuple[str, int]] = field(default_factory=list)
    default_floors: bool = True
    ple_bits: int = 4
    min_bits: int = 3
    sens_power: float = 1.0
    score: str = "reap"
    domain_weights: Optional[Dict[str, float]] = None
    min_keep_frac: float = 0.25
    auto_step: Optional[int] = None
    prune_lambda: float = 1.0
    mtp_sens_weight: float = 0.25
    dtype: Optional[str] = None
    shard_gb: float = 2.0
    dry_run: bool = False
    rounding: str = "rtn"  # rtn | lsq (docs/plan/05 §2.6a)
    lsq_iters: int = 4


DEFAULT_FLOORS = ((r"lm_head$", 6), (r"embed_tokens$", 4))


# ----------------------------------------------------------------------------- chunks


@dataclass
class Item:
    out_key: str
    keys: List[str]
    rows: Optional[List[int]] = None
    stack: bool = False


def materialize(src: Source, items: List[Item], real: bool) -> Dict[str, mx.array]:
    get = src.read if real else src.placeholder
    out = {}
    for it in items:
        if it.stack:
            if real:
                parts = []
                for k in it.keys:
                    parts.append(src.read(k))
                    if len(parts) % 16 == 0:
                        mx.eval(parts[-16:])
                out[it.out_key] = mx.stack(parts)
                del parts
            else:
                first = src.placeholder(it.keys[0])
                out[it.out_key] = mx.zeros((len(it.keys), *first.shape), dtype=first.dtype)
        else:
            out[it.out_key] = get(it.keys[0], it.rows)
    return out


def layer_items(
    layer: MoELayer, keep: Optional[List[int]], src: Source
) -> Tuple[List[Item], List[List[Item]]]:
    """(router items, expert chunks) for one MoE block under ``keep``."""
    keep = list(range(layer.num_experts)) if keep is None else keep
    router = []
    for key in layer.router_keys:
        shape = src.info(key).shape
        rows = keep if shape and shape[0] == layer.num_experts else None
        router.append(Item(key, [key], rows))
    # fused tensors share one chunk: mlx-vlm moves down_proj only alongside gate_up_proj
    chunks = [[Item(key, [key], keep) for key in layer.fused_keys]] if layer.fused_keys else []
    for proj, experts in sorted(layer.per_expert.items()):
        missing = [e for e in keep if e not in experts]
        if missing:
            raise ValueError(f"layer {layer.name}: missing {proj} for experts {missing[:8]}")
        chunks.append([Item(f"{layer.prefix}mlp.switch_mlp.{proj}.weight", [experts[e] for e in keep], stack=True)])
    return router, chunks


# ----------------------------------------------------------------------------- models


def prepare_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("text_config", cfg.pop("llm_config", {}))
    cfg.setdefault("vision_config", {})
    cfg.setdefault("audio_config", {})
    return cfg


@dataclass
class Target:
    """An instantiated (lazy, never evaluated) mlx model + how to turn source tensors into its weights."""

    model: object
    transform: Callable[[Dict[str, mx.array]], Dict[str, mx.array]]
    quant_predicate: Optional[Callable]
    cast_predicate: Callable[[str], bool]
    modules: Dict[str, object]

    def params(self) -> Dict[str, Tuple[Tuple[int, ...], object]]:
        return {k: (tuple(v.shape), v.dtype) for k, v in tree_flatten(self.model.parameters())}


def main_target(family: Family, cfg: dict) -> Target:
    cfg = prepare_config(cfg)
    if cfg.get("model_file"):
        arch = shim.load_module(shim.render(family.model_type, family.experts_attr))
    else:
        arch = importlib.import_module(f"mlx_vlm.models.{family.model_type}")
    model_config = arch.ModelConfig.from_dict(cfg)
    model_config = update_module_configs(model_config, arch, cfg, ["text", "vision", "perceiver", "projector", "audio"])
    model = arch.Model(model_config)
    fns = [model.sanitize]
    if hasattr(getattr(arch, "VisionModel", None), "sanitize"):
        fns.append(arch.VisionModel(model_config.vision_config).sanitize)
    if hasattr(getattr(arch, "LanguageModel", None), "sanitize"):
        fns.append(arch.LanguageModel(model_config.text_config).sanitize)

    def transform(weights):
        for fn in fns:
            weights = fn(weights)
        return weights

    model_pred = getattr(model, "quant_predicate", None)

    def quant_predicate(path, module):
        if skip_multimodal_module(path):
            return False
        return model_pred(path, module) if model_pred is not None else True

    return Target(
        model,
        transform,
        quant_predicate,
        getattr(model, "cast_predicate", lambda _k: True),
        dict(model.named_modules()),
    )


def drafter_config(family: Family, text_config: dict, num_experts: int) -> dict:
    splitter = family.drafter()
    tc = copy.deepcopy(text_config)
    tc[family.experts_attr] = int(num_experts)
    tc.pop("quantization_config", None)
    cfg = {
        "model_type": splitter.output_model_type,
        "text_config": tc,
        "block_size": int(splitter.depth(tc) + splitter.block_size_extra),
        "tie_word_embeddings": bool(tc.get("tie_word_embeddings", splitter.tie_word_embeddings_default)),
    }
    cfg.update(splitter.extra_config(tc))
    return cfg


def mtp_target(family: Family, draft_cfg: dict, cast_predicate) -> Target:
    splitter = family.drafter()
    arch = importlib.import_module(f"mlx_vlm.speculative.drafters.{draft_cfg['model_type']}")
    model = arch.Model(arch.ModelConfig.from_dict(copy.deepcopy(draft_cfg)))
    tc = draft_cfg["text_config"]

    def transform(weights):
        return splitter.transform(weights, tc, source_is_mlx=False)

    return Target(
        model,
        transform,
        getattr(model, "quant_predicate", None),
        cast_predicate,
        dict(model.named_modules()),
    )


# ----------------------------------------------------------------------------- plan


@dataclass
class Tensor:
    ns: str
    key: str
    shape: Tuple[int, ...]
    itemsize: int
    layer: Optional[str]
    expert_axis: bool

    def nbytes(self, counts: Dict[str, int]) -> int:
        shape = self.shape
        if self.expert_axis and self.layer in counts:
            shape = (counts[self.layer], *shape[1:])
        return int(np.prod(shape)) * self.itemsize


class Builder:
    def __init__(self, opts: BuildOptions):
        self.opts = opts
        self.src = Source(opts.src)
        self.config = self.src.config
        self.family = family_for(opts.model or self.config.get("model_type"))
        if self.config.get("model_type") != self.family.model_type:
            raise ValueError(
                f"--model {opts.model} expects model_type {self.family.model_type}, "
                f"source has {self.config.get('model_type')}"
            )
        self.text_config = self.config["text_config"]
        self.num_layers = int(self.text_config["num_hidden_layers"])
        self.E = int(self.text_config[self.family.experts_attr])
        self.top_k = int(self.text_config.get("num_experts_per_tok", 1))
        dtype_name = opts.dtype or self.text_config.get("dtype") or self.config.get("torch_dtype") or "bfloat16"
        self.dtype = DTYPES.get(dtype_name, mx.bfloat16)
        keys = self.src.keys()
        self.moe = find_moe_layers(keys, self.family, self.text_config)
        self.main_moe = {n: l for n, l in self.moe.items() if n != "mtp"}
        self.mtp_keys = [k for k in keys if self.family.is_mtp_key(k, self.text_config)]
        self.has_mtp = bool(self.mtp_keys) and opts.mtp != "drop"
        self.main_keys = [
            k
            for k in keys
            if not self.family.is_mtp_key(k, self.text_config)
            and not (opts.drop_vision and k.startswith(self.family.vision_prefix))
        ]
        self.calib = CalibStats.load(opts.calib)
        self.scores: Dict[str, prune.ExpertScores] = {}
        self.timings: Dict[str, float] = {}
        self.sens_report: Dict[str, object] = {}
        self.round_stats: Dict[str, Dict[str, float]] = {}
        self._imatrix_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # -- expert scores / keep plans

    def compute_scores(self):
        names = list(self.main_moe) + (["mtp"] if self.has_mtp and "mtp" in self.moe else [])
        keep_all = prune.parse_prune(self.opts.prune, self.E) == ("keep", self.E)
        for name in names:
            if keep_all and not self.calib.get(name, (str(self.num_layers), "mtp") if name == "mtp" else ()):
                ones = np.ones(self.E)
                self.scores[name] = prune.ExpertScores(ones, ones / self.E, None, "none")
                continue
            aliases = (str(self.num_layers), "mtp") if name == "mtp" else ()
            stats = self.calib.get(name, aliases)
            s = prune.calib_scores(stats, self.E, self.opts.score, self.opts.domain_weights)
            if s is None:
                if name != "mtp" and self.opts.prune not in ("none", "0"):
                    log.warning("no expert stats for layer %s in calib; using weight saliency", name)
                s = prune.weight_scores(self.src, self.moe[name])
            self.scores[name] = s

    def min_keep(self) -> int:
        return max(self.top_k, int(np.ceil(self.opts.min_keep_frac * self.E)))

    def plan_for(self, k: Optional[int] = None, total: Optional[int] = None) -> Dict[str, List[int]]:
        main_scores = {n: self.scores[n] for n in self.main_moe}
        if total is not None:
            plan = prune.global_keep(main_scores, total, {n: self.min_keep() for n in main_scores})
            mtp_k = int(round(total / max(len(main_scores), 1)))
        else:
            plan = prune.uniform_keep(main_scores, k)
            mtp_k = k
        if self.has_mtp and "mtp" in self.moe:
            if self.opts.mtp == "intact":
                plan["mtp"] = list(range(self.E))
            else:
                plan["mtp"] = prune.keep_top(self.scores["mtp"], min(max(mtp_k, self.top_k), self.E))
        return plan

    # -- configs

    def counts(self, plan) -> Dict[str, int]:
        return {n: len(v) for n, v in plan.items()}

    def out_config(self, plan) -> dict:
        cfg = copy.deepcopy(self.config)
        for k in ("quantization_config", "quantization"):
            cfg.pop(k, None)
            cfg["text_config"].pop(k, None)
        tc = cfg["text_config"]
        per_layer = [len(plan[str(i)]) if str(i) in plan else None for i in range(self.num_layers)]
        distinct = sorted({c for c in per_layer if c})
        if len(distinct) <= 1:
            tc[self.family.experts_attr] = distinct[0] if distinct else self.E
        else:
            tc[self.family.experts_attr] = max(distinct)
            cfg["expert_counts"] = per_layer
            cfg["model_file"] = shim.SHIM_FILENAME
        return cfg

    # -- chunking

    def main_chunks(self, plan) -> List[List[Item]]:
        expert_owned = {}
        for layer in self.main_moe.values():
            for k in layer.keys:
                expert_owned[k] = layer
        per_layer: Dict[int, List[Item]] = OrderedDict()
        singles: List[List[Item]] = []
        vision: List[Item] = []
        for key in self.main_keys:
            if key in expert_owned:
                continue
            idx = self.family.layer_of(key)
            if ".ngram_embedding." in key or idx is None and not key.startswith(self.family.vision_prefix):
                singles.append([Item(key, [key])])
            elif idx is not None:
                per_layer.setdefault(idx, []).append(Item(key, [key]))
            else:
                vision.append(Item(key, [key]))
        chunks = list(singles)
        if vision:
            chunks.append(vision)
        for idx, items in per_layer.items():
            layer = self.main_moe.get(str(idx))
            if layer is None:
                chunks.append(items)
                continue
            router, expert_chunks = layer_items(layer, plan.get(layer.name), self.src)
            chunks.append(items + router)
            chunks.extend(expert_chunks)
        return chunks

    def mtp_chunks(self, plan) -> List[List[Item]]:
        layer = self.moe.get("mtp")
        owned = set(layer.keys) if layer else set()
        rest = [Item(k, [k]) for k in self.mtp_keys if k not in owned]
        if layer is None:
            return [rest]
        router, expert_chunks = layer_items(layer, plan.get("mtp"), self.src)
        return [rest + router] + expert_chunks

    # -- tensor processing

    def keep_dtype(self, key: str, value: mx.array, target: Target) -> bool:
        if not mx.issubdtype(value.dtype, mx.floating):
            return True
        if ROUTER_KEEP_DTYPE.search(key) and value.dtype == mx.float32:
            return True
        return not target.cast_predicate(key)

    def process(self, items, target: Target, real: bool) -> Dict[str, mx.array]:
        weights = target.transform(materialize(self.src, items, real))
        return {
            k: (v if self.keep_dtype(k, v, target) else v.astype(self.dtype)) for k, v in weights.items()
        }

    def dry_run(self, chunks, target: Target, ns: str, moe_prefix: Callable[[str], Optional[str]], counts):
        produced: Dict[str, Tuple[Tuple[int, ...], object]] = {}
        for items in chunks:
            for k, v in self.process(items, target, real=False).items():
                if k in produced:
                    raise ValueError(f"[{ns}] tensor {k} produced twice")
                produced[k] = (tuple(v.shape), v.dtype)
        expected = target.params()
        missing = [
            k
            for k in expected
            if k not in produced and not (self.opts.drop_vision and k.startswith("vision_tower"))
        ]
        extra = [k for k in produced if k not in expected]
        bad = [(k, produced[k][0], expected[k][0]) for k in produced if k in expected and produced[k][0] != expected[k][0]]
        if missing or extra or bad:
            raise ValueError(
                f"[{ns}] planned weights do not match the mlx-vlm model: "
                f"missing={missing[:10]} extra={extra[:10]} shape_mismatch={bad[:10]}"
            )
        tensors = []
        for k, (shape, dtype) in produced.items():
            layer = moe_prefix(k)
            expert_axis = (
                layer is not None
                and (".switch_mlp." in k or ".mlp.gate." in k)
                and len(shape) > 0
                and shape[0] == counts.get(layer, -1)
            )
            tensors.append(Tensor(ns, k, shape, dtype.size, layer, expert_axis))
        return tensors

    def main_moe_layer_of(self, key: str) -> Optional[str]:
        m = re.match(r"^language_model\.model\.layers\.(\d+)\.mlp\.", key)
        return m.group(1) if m and m.group(1) in self.main_moe else None

    @staticmethod
    def mtp_moe_layer_of(key: str) -> Optional[str]:
        return "mtp" if key.startswith(("layers.0.mlp.", "mtp_block.mlp.")) else None

    # -- units

    def make_units(self, tensors: List[Tensor], target: Target, ns: str) -> Tuple[List[alloc.Unit], List[Tensor]]:
        units, fixed = [], []
        pins = [(re.compile(p), b, g) for p, b, g in list(self.family.default_pins) + list(self.opts.pins)]
        floors = [(re.compile(p), b) for p, b in (DEFAULT_FLOORS if self.opts.default_floors else ())]
        floors += [(re.compile(p), b) for p, b in self.opts.floors]
        for t in tensors:
            path = t.key[: -len(".weight")] if t.key.endswith(".weight") else None
            module = target.modules.get(path) if path else None
            ok = module is not None and hasattr(module, "to_quantized")
            pred = target.quant_predicate(path, module) if ok and target.quant_predicate else ok
            groups = tuple(g for g in self.opts.groups if t.shape[-1] % g == 0)
            if not ok or pred is False or not groups or len(t.shape) < 2:
                fixed.append(t)
                continue
            kind = alloc.classify(path, self.family.ple_marker)
            unit = alloc.Unit(
                ns=ns,
                path=path,
                shape=t.shape,
                itemsize=t.itemsize,
                kind=kind,
                layer=t.layer,
                expert_axis=t.expert_axis,
                allowed_groups=groups,
                floor=self.opts.min_bits,
            )
            if kind == "ple_table":
                unit.pinned = (self.opts.ple_bits, 32 if 32 in groups else groups[0])
            for rx, b, g in pins:
                if rx.search(path):
                    unit.pinned = (b, g)
            for rx, b in floors:
                if rx.search(path):
                    unit.floor = max(unit.floor, b)
            units.append(unit)
        return units, fixed

    # -- sensitivity assignment

    def resolve_sens_name(self, layer_ctx: str, raw: str) -> List[Tuple[Tuple[str, str], Optional[int]]]:
        name = raw.strip()
        name = re.sub(r"\.(weight|qweight)$", "", name)
        fam = self.family
        is_mtp = layer_ctx == "mtp" or fam.is_mtp_key(name + ".", self.text_config) or name.startswith("mtp.")
        if layer_ctx == str(self.num_layers) and fam.name == "glm":
            is_mtp = True
        layer_rel = name.startswith(("mlp.", "self_attn.", "linear_attn.", "attn_hyper", "mlp_hyper", "ple.", "input_layernorm", "post_attention"))
        if is_mtp:
            ns = "mtp"
            if fam.name == "qwen":
                for p in ("model.mtp.", "language_model.mtp.", "mtp."):
                    if name.startswith(p):
                        name = name[len(p) :]
                if layer_rel:
                    name = "layers.0." + name
            else:
                pre = fam.mtp_layer_prefix(self.text_config)
                if name.startswith(pre):
                    name = name[len(pre) :]
                    layer_rel = not name.startswith(("enorm", "hnorm", "eh_proj", "shared_head"))
                if name.startswith("shared_head.norm"):
                    name = "shared_head_norm" + name[len("shared_head.norm") :]
                elif layer_rel and not name.startswith("mtp_block."):
                    name = "mtp_block." + name
        else:
            ns = "main"
            if layer_rel and layer_ctx.isdigit():
                name = f"language_model.model.layers.{layer_ctx}." + name
            elif name.startswith("layers."):
                name = "language_model.model." + name
            name = fam.mlx_prefix(name)
        expert = None
        m = re.search(r"\.mlp\.experts\.(\d+)\.(\w+)$", name)
        if m:
            expert = int(m.group(1))
            name = name[: m.start()] + ".mlp.switch_mlp." + m.group(2)
        name = re.sub(r"\.shard_(\d+)$", r".shards.\1", name)
        candidates = [name]
        for srcs, dsts in fam.fusions:
            for s in srcs:
                if name.endswith("." + s) or name == s:
                    base = name[: len(name) - len(s)]
                    candidates = [base + d for d in dsts]
                    break
            else:
                continue
            break
        return [((ns, c), expert) for c in candidates]

    def assign_sens(self, units: List[alloc.Unit]):
        by_key = {u.key: u for u in units}
        by_suffix: Dict[Tuple[str, str], List[alloc.Unit]] = {}
        unresolved = []
        n_entries = 0
        for layer_ctx, raw, grid in self.calib.sens_items():
            n_entries += 1
            hits = self.resolve_sens_name(layer_ctx, raw)
            matched = False
            for (ns, path), expert in hits:
                unit = by_key.get((ns, path))
                if unit is None:
                    suffix = path.split("layers.")[-1] if "layers." in path else path
                    cands = by_suffix.get((ns, suffix))
                    if cands is None:
                        cands = [u for u in units if u.ns == ns and u.path.endswith(suffix)]
                        by_suffix[(ns, suffix)] = cands
                    unit = cands[0] if len(cands) == 1 else None
                if unit is None:
                    continue
                matched = True
                if expert is not None:
                    unit.expert_sens.setdefault(expert, {}).update(grid)
                else:
                    unit.sens.update(grid)
                unit.sens_source = "calib"
            if not matched:
                unresolved.append(f"{layer_ctx}:{raw}")
        if unresolved:
            log.warning(
                "%d calib sensitivity entries did not match any quantizable tensor (e.g. %s)",
                len(unresolved),
                unresolved[:5],
            )
        for u in units:
            if u.expert_sens and not u.sens:
                freq = self.scores[u.layer].freq if u.layer in self.scores else None
                u.sens = dict(alloc.aggregate_expert_sens(u, None, freq))
        unparsed = sum(len(st.unparsed) for st in self.calib.layers.values())
        matched = sum(u.sens_source == "calib" for u in units)
        self.sens_report = {"entries": n_entries, "unmatched": len(unresolved), "unparsed": unparsed,
                            "units_calibrated": matched, "units": len(units)}
        if self.opts.calib and (n_entries or unparsed) and not matched:
            raise ValueError(
                f"none of the {n_entries} parsed calib sensitivity entries ({unparsed} unparsed) in {self.opts.calib} "
                f"match a quantizable tensor (e.g. {unresolved[:3]})"
            )

    # -- allocation

    def options_for(self, units: List[alloc.Unit], plan) -> List[List[alloc.Option]]:
        out = []
        for u in units:
            k = len(plan[u.layer]) if u.expert_axis and u.layer in plan else None
            shape = u.with_experts(k) if k else u.shape
            sens = u.sens
            if u.expert_sens:
                freq = self.scores.get(u.layer).freq if u.layer in self.scores else None
                agg = alloc.aggregate_expert_sens(u, plan.get(u.layer), freq)
                sens = {**u.sens, **agg} if agg else u.sens
            weight = self.opts.mtp_sens_weight if u.ns == "mtp" else 1.0
            sens = {c: v**self.opts.sens_power for c, v in sens.items()}
            out.append(alloc.unit_options(u, shape, sens, self.opts.bits, self.opts.groups, self.opts.allow_bf16, weight))
        return out

    def fixed_bytes(self, fixed: List[Tensor], plan) -> int:
        counts = self.counts(plan)
        return sum(t.nbytes(counts) for t in fixed)

    def allocate(self, units, fixed, plan, method):
        budget = int(self.opts.budget_gb * GB)
        options = self.options_for(units, plan)
        fixed_b = self.fixed_bytes(fixed, plan)
        sol = alloc.solve(options, budget - fixed_b, method)
        return sol, fixed_b, options

    def total_prune_loss(self, plan, power: float = 1.0) -> float:
        return sum(prune.prune_loss(self.scores[n], plan[n]) ** power for n in self.main_moe if n in plan)

    def search_auto(self, units, fixed) -> Tuple[Dict[str, List[int]], List[dict]]:
        step = self.opts.auto_step or max(8, self.E // 32)
        lo = self.min_keep()
        grid = sorted({min(self.E, max(lo, k)) for k in range(step * int(np.ceil(lo / step)), self.E + 1, step)} | {self.E})
        trace = []
        best = None
        n_layers = len(self.main_moe)
        for k in grid:
            plan = self.plan_for(total=k * n_layers) if self.opts.prune_mode == "global" else self.plan_for(k=k)
            sol, fixed_b, _ = self.allocate(units, fixed, plan, "greedy")
            if sol is None:
                trace.append({"keep": k, "feasible": False})
                continue
            loss = self.total_prune_loss(plan, self.opts.sens_power)
            j = sol.objective + self.opts.prune_lambda * loss
            trace.append(
                {"keep": k, "feasible": True, "quant_obj": sol.objective, "prune_loss": loss, "objective": j,
                 "bytes": sol.bytes + fixed_b}
            )
            if best is None or j < best[0]:
                best = (j, plan)
        if best is None:
            raise ValueError(f"no expert count fits the {self.opts.budget_gb} GB budget")
        return best[1], trace

    # -- top level

    def run(self) -> dict:
        t0 = time.time()
        opts = self.opts
        mx.set_cache_limit(1 << 30)
        self.compute_scores()
        full_plan = {n: list(range(self.E)) for n in self.moe if n != "mtp" or self.has_mtp}

        mode, k = prune.parse_prune(opts.prune, self.E, self.top_k)
        main_cfg_full = self.out_config(full_plan)
        main_full = main_target(self.family, main_cfg_full)
        tensors = self.dry_run(self.main_chunks(full_plan), main_full, "main", self.main_moe_layer_of, self.counts(full_plan))
        units, fixed = self.make_units(tensors, main_full, "main")
        if self.has_mtp:
            dcfg = drafter_config(self.family, self.text_config, self.E)
            mt = mtp_target(self.family, dcfg, main_full.cast_predicate)
            mtp_tensors = self.dry_run(self.mtp_chunks(full_plan), mt, "mtp", self.mtp_moe_layer_of, self.counts(full_plan))
            mu, mf = self.make_units(mtp_tensors, mt, "mtp")
            units += mu
            fixed += mf
        del main_full
        self.assign_sens(units)
        alloc.fill_sensitivities(units, opts.bits, opts.groups)
        self.timings["plan"] = time.time() - t0

        trace = None
        if mode == "auto":
            plan, trace = self.search_auto(units, fixed)
        elif opts.prune_mode == "global" and k < self.E:
            plan = self.plan_for(total=k * len(self.main_moe))
        else:
            plan = self.plan_for(k=k)

        sol, fixed_b, options = self.allocate(units, fixed, plan, opts.solver)
        if sol is None:
            minimum = self.fixed_bytes(fixed, plan) + sum(min(o.bytes for o in opts_) for opts_ in self.options_for(units, plan))
            raise ValueError(
                f"budget {opts.budget_gb} GB is infeasible for this plan; the smallest possible build is "
                f"{minimum / GB:.2f} GB (prune more, drop vision/MTP, or lower --floor/--ple-bits)"
            )
        for u, c in zip(units, sol.choices):
            u.choice = c
        self.timings["solve"] = time.time() - t0 - self.timings["plan"]

        out = Path(opts.out)
        cfg = self.out_config(plan)
        main_t = main_target(self.family, cfg)
        final_main = self.dry_run(self.main_chunks(plan), main_t, "main", self.main_moe_layer_of, self.counts(plan))
        main_units = {u.path: u for u in units if u.ns == "main"}
        quant_cfg = self.quant_config(main_units.values())
        cfg["quantization"] = quant_cfg
        cfg["quantization_config"] = copy.deepcopy(quant_cfg)
        predicted = {"main": self.predict_bytes(final_main, main_units)}

        mtp_info = None
        if self.has_mtp:
            k_mtp = len(plan["mtp"]) if "mtp" in plan else self.E
            dcfg = drafter_config(self.family, self.text_config, k_mtp)
            mt = mtp_target(self.family, dcfg, main_t.cast_predicate)
            final_mtp = self.dry_run(self.mtp_chunks(plan), mt, "mtp", self.mtp_moe_layer_of, self.counts(plan))
            mtp_units = {u.path: u for u in units if u.ns == "mtp"}
            mq = self.quant_config(mtp_units.values())
            dcfg["quantization"] = mq
            dcfg["quantization_config"] = copy.deepcopy(mq)
            predicted["mtp"] = self.predict_bytes(final_mtp, mtp_units)
            mtp_info = (dcfg, mt, mtp_units)

        if sum(predicted.values()) != sol.bytes + fixed_b:
            raise RuntimeError(
                f"size model mismatch: planned {sol.bytes + fixed_b} vs final tensors {sum(predicted.values())}"
            )
        summary = self.summary(plan, units, sol, fixed_b, predicted, trace)
        if opts.dry_run:
            summary["dry_run"] = True
            self.timings["total"] = time.time() - t0
            summary["timings_s"] = self.timings
            return summary

        out.mkdir(parents=True, exist_ok=True)
        if not self.has_mtp and (out / "mtp").is_dir():
            shutil.rmtree(out / "mtp")
        if not cfg.get("model_file") and (out / shim.SHIM_FILENAME).exists():
            (out / shim.SHIM_FILENAME).unlink()
        t1 = time.time()
        written = {"main": self.write(self.main_chunks(plan), main_t, main_units, out, plan=plan)}
        if cfg.get("model_file"):
            (out / shim.SHIM_FILENAME).write_text(shim.render(self.family.model_type, self.family.experts_attr))
        self.copy_aux(out)
        save_json(out / "config.json", dict(sorted(cfg.items())))
        if mtp_info is not None:
            dcfg, mt, mtp_units = mtp_info
            mdir = out / "mtp"
            mdir.mkdir(exist_ok=True)
            written["mtp"] = self.write(self.mtp_chunks(plan), mt, mtp_units, mdir)  # no imatrix: own layer indices
            save_json(mdir / "config.json", dict(sorted(dcfg.items())))
            self.copy_aux(mdir)
        self.timings["write"] = time.time() - t1
        self.timings["total"] = time.time() - t0
        summary["written_bytes"] = written
        summary["rounding"] = {"method": self.opts.rounding, "lsq_iters": self.opts.lsq_iters, **{
            k: {**v, "ratio": v["err_lsq"] / v["err_rtn"] if v.get("err_rtn") else None}
            for k, v in sorted(self.round_stats.items())}}
        summary["timings_s"] = self.timings
        for ns, nbytes in written.items():
            if nbytes != predicted[ns]:
                raise RuntimeError(f"[{ns}] size accounting mismatch: predicted {predicted[ns]} wrote {nbytes}")
        save_json(out / "build_summary.json", summary)
        return summary

    # -- output helpers

    @staticmethod
    def quant_config(units) -> dict:
        chosen = [u.choice for u in units if u.choice.bits != alloc.BF16]
        common = Counter((c.bits, c.group) for c in chosen).most_common(1)
        bits, group = common[0][0] if common else (4, 64)
        cfg = {"group_size": group, "bits": bits, "mode": "affine"}
        for u in sorted(units, key=lambda u: u.path):
            c = u.choice
            cfg[u.path] = False if c.bits == alloc.BF16 else {"group_size": c.group, "bits": c.bits, "mode": "affine"}
        return cfg

    @staticmethod
    def predict_bytes(tensors: List[Tensor], units: Dict[str, alloc.Unit]) -> int:
        total = 0
        for t in tensors:
            path = t.key[: -len(".weight")] if t.key.endswith(".weight") else None
            u = units.get(path)
            if u is not None and u.choice.bits != alloc.BF16:
                numel = int(np.prod(t.shape))
                total += alloc.quantized_bytes(numel, t.shape[-1], u.choice.bits, u.choice.group, t.itemsize)
            else:
                total += int(np.prod(t.shape)) * t.itemsize
        return total

    _LAYER_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")

    def imatrix(self, path: str, shape, plan: Optional[Dict[str, List[int]]]) -> Optional[mx.array]:
        """calib ``act_sqmean`` for this tensor's input channels: [in] or, for stacked experts, [kept, in]."""
        m = self._LAYER_PATH.search(path)
        if plan is None or not self.opts.calib or m is None:
            return None
        i, name = int(m.group(1)), m.group(2)
        if i not in self._imatrix_cache:
            f = Path(self.opts.calib) / f"layer_{i:03d}.npz"
            arrays = {}
            if f.exists():
                with np.load(f, allow_pickle=False) as z:
                    arrays = {k[len("act_sqmean."):]: z[k] for k in z.files if k.startswith("act_sqmean.")}
            self._imatrix_cache = {i: arrays}  # layers are written in order; keep one
        h = self._imatrix_cache[i].get(name)
        if h is None:
            return None
        if h.ndim == 2:
            kept = plan.get(str(i))
            if kept is not None and len(kept) != h.shape[0]:
                h = h[np.asarray(kept)]
            if len(shape) != 3 or h.shape[0] != shape[0]:
                return None
        return mx.array(h.astype(np.float32))

    def write(self, chunks, target: Target, units: Dict[str, alloc.Unit], out: Path, plan=None) -> int:
        writer = ShardWriter(out, int(self.opts.shard_gb * GB))
        for i, items in enumerate(chunks):
            weights = self.process(items, target, real=True)
            result = {}
            for k, v in weights.items():
                path = k[: -len(".weight")] if k.endswith(".weight") else None
                u = units.get(path)
                if u is not None and u.choice.bits != alloc.BF16:
                    h = self.imatrix(path, v.shape, plan) if self.opts.rounding != "rtn" else None
                    st = self.round_stats.setdefault(f"{u.kind}:{u.choice.label}", {})
                    q = rounding.quantize(v, u.choice.bits, u.choice.group, self.opts.rounding, h=h,
                                          iters=self.opts.lsq_iters, stats=st)
                    st["weighted"] = st.get("weighted", 0) + (h is not None)
                    st["tensors"] = st.get("tensors", 0) + 1
                    result[k] = q[0]
                    result[f"{path}.scales"] = q[1]
                    result[f"{path}.biases"] = q[2]
                else:
                    result[k] = v
            del weights
            writer.add(result)
            del result
            mx.clear_cache()
            if (i + 1) % 25 == 0:
                log.info("chunk %d/%d, %.2f GB written", i + 1, len(chunks), writer.total / GB)
        return writer.finish()

    def copy_aux(self, out: Path):
        for pattern in TOKENIZER_GLOBS:
            for f in self.src.path.glob(pattern):
                if f.name in SKIP_COPY or f.name.endswith(".safetensors.index.json"):
                    continue
                shutil.copy2(f, out / f.name)
        if not (out / "generation_config.json").exists():
            eos = self.config.get("eos_token_id", self.text_config.get("eos_token_id"))
            save_json(out / "generation_config.json", {"eos_token_id": eos})

    def summary(self, plan, units, sol, fixed_b, predicted, trace) -> dict:
        cats: Dict[str, int] = Counter()
        for u, c in zip(units, sol.choices):
            cats[f"{u.ns}:{u.kind}"] += c.bytes
        hist = Counter()
        params = Counter()
        for u, c in zip(units, sol.choices):
            hist[c.label] += 1
            k = len(plan[u.layer]) if u.expert_axis and u.layer in plan else None
            params[c.label] += int(np.prod(u.with_experts(k) if k else u.shape))
        q_params = sum(params.values())
        q_bytes = sum(c.bytes for c in sol.choices)
        per_layer = {
            n: {
                "kept": len(plan[n]),
                "of": self.E,
                "prune_loss": round(prune.prune_loss(self.scores[n], plan[n]), 6),
                "score_source": self.scores[n].source,
                "kept_ids": plan[n],
            }
            for n in plan
            if n in self.scores
        }
        return {
            "model": self.family.name,
            "model_type": self.family.model_type,
            "source": str(self.src.path),
            "calib": str(self.opts.calib) if self.opts.calib else None,
            "budget_bytes": int(self.opts.budget_gb * GB),
            "predicted_bytes": predicted,
            "predicted_total_bytes": sum(predicted.values()),
            "fixed_bytes": fixed_b,
            "quantized_bytes": q_bytes,
            "quantized_params": q_params,
            "avg_bits_per_weight": round(8 * q_bytes / max(q_params, 1), 4),
            "objective": sol.objective,
            "solver": sol.method,
            "sens_power": self.opts.sens_power,
            "calib_sens": self.sens_report,
            "prune": {
                "arg": self.opts.prune,
                "mode": self.opts.prune_mode,
                "experts_per_layer": {n: len(v) for n, v in plan.items()},
                "total_prune_loss": round(self.total_prune_loss(plan), 6),
                "layers": per_layer,
                "auto_trace": trace,
            },
            "mtp": None if not self.has_mtp else {"mode": self.opts.mtp, "path": "mtp", "experts": len(plan.get("mtp", [])) or self.E},
            "drop_vision": self.opts.drop_vision,
            "bits_histogram_units": dict(hist),
            "bits_histogram_params": dict(params),
            "bytes_by_kind": dict(cats),
            "allocation": [
                {
                    "ns": u.ns,
                    "path": u.path,
                    "kind": u.kind,
                    "shape": list(u.with_experts(len(plan[u.layer])) if u.expert_axis and u.layer in plan else u.shape),
                    "bits": c.bits,
                    "group_size": c.group,
                    "bytes": c.bytes,
                    "cost": c.sens,
                    "sens_source": u.sens_source,
                    "pinned": u.pinned is not None,
                }
                for u, c in zip(units, sol.choices)
            ],
        }


class ShardWriter:
    def __init__(self, out: Path, max_bytes: int):
        self.out, self.max_bytes = out, max_bytes
        self.pending: Dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.files: List[Tuple[Path, List[str]]] = []
        self.total = 0
        for old in list(out.glob("model*.safetensors")) + list(out.glob(".tmp-*.safetensors")):
            old.unlink()

    def add(self, weights: Dict[str, mx.array]):
        mx.eval(list(weights.values()))
        for k, v in weights.items():
            self.pending[k] = v
            self.pending_bytes += v.nbytes
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        path = self.out / f".tmp-{len(self.files):05d}.safetensors"
        mx.save_safetensors(str(path), self.pending, metadata={"format": "mlx"})
        self.files.append((path, sorted(self.pending)))
        self.total += self.pending_bytes
        self.pending, self.pending_bytes = {}, 0
        mx.clear_cache()

    def finish(self) -> int:
        self.flush()
        n = len(self.files)
        weight_map = {}
        for i, (path, keys) in enumerate(self.files):
            name = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            path.rename(self.out / name)
            for k in keys:
                weight_map[k] = name
        save_json(
            self.out / "model.safetensors.index.json",
            {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(weight_map.items()))},
        )
        return self.total


def save_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
