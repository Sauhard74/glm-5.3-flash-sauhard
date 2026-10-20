"""Pins one model into mlx_vlm.server and enforces this machine's co-tenant rules.

mlx-vlm's server already provides the OpenAI/Anthropic protocol layers, native
tool-call parsing, the automatic prefix cache (with checkpoint restore for
hybrid linear-attention caches) and MTP speculative decoding. This module wraps
it: one pinned model whatever name a client asks for, a hard memory cap, a
context-aware max_tokens clamp, single-sequence decoding, and idle unload.
"""

import contextlib
import contextvars
import functools
import gc
import importlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx

from . import memory
from .config import CLEARED_ENV, GB, ServeConfig

logger = logging.getLogger("serve.engine")

_request_args = contextvars.ContextVar("serve_request_args", default=None)
_ACTIVE: Optional["Engine"] = None
_ORIG: dict = {}


def _patch_mlx_vlm():
    """Patch mlx_vlm.server once per process; every hook dispatches to _ACTIVE."""
    if _ORIG:
        return
    import mlx_vlm.server as srv
    from mlx_vlm import apc
    from mlx_vlm import generate as gen_pkg
    from mlx_vlm.generate import ar, common, diffusion, dispatch
    from mlx_vlm.models import cache as cache_mod
    from mlx_vlm.server import generation as gen
    from mlx_vlm.server import request_normalization as rn

    app_mod = importlib.import_module("mlx_vlm.server.app")
    RG = gen.ResponseGenerator
    _ORIG.update(
        batch_extract=cache_mod.BatchKVCache.extract,
        wired_limit=common.wired_limit,
        get_cached_model=app_mod.get_cached_model,
        working_set=apc._metal_working_set_bytes,
        check=gen._check_configured_context_budget,
        generate=RG.generate,
        validate=RG.validate_context_budget,
        step=RG._step,
        build=rn._build_gen_args,
    )

    def get_cached_model(*a, **kw):
        if _ACTIVE is None:
            return _ORIG["get_cached_model"](*a, **kw)
        return _ACTIVE.get_cached_model(*a, **kw)

    # APC sizes its resident tier against the Metal working set; use our cap.
    def working_set():
        return _ACTIVE.cfg.memory_limit_bytes if _ACTIVE else _ORIG["working_set"]()

    def check_context(prompt_tokens: int, max_tokens: int):
        limit = gen.get_configured_context_limit()
        args = _request_args.get()
        if limit and args is not None and prompt_tokens < limit:
            room = limit - prompt_tokens
            if max_tokens > room:
                args.max_tokens = max_tokens = room
        return _ORIG["check"](prompt_tokens, max_tokens)

    def with_args(method):
        @functools.wraps(method)
        def wrapper(self_, prompt, images=None, audio=None, args=None, videos=None):
            token = _request_args.set(args)
            try:
                return method(self_, prompt, images=images, audio=audio, args=args, videos=videos)
            finally:
                _request_args.reset(token)

        return wrapper

    def guarded_step(self_, batch_gen, active, gen_kwargs=None):
        if _ACTIVE is not None:
            _ACTIVE.guard.enforce(getattr(self_, "apc_manager", None))
        return _ORIG["step"](self_, batch_gen, active, gen_kwargs)

    def build_gen_args(*a, **kw):
        args = _ORIG["build"](*a, **kw)
        if _ACTIVE is not None:
            args.max_tokens = max(1, min(int(args.max_tokens), _ACTIVE.cfg.max_output_tokens))
            if _ACTIVE.mtp_active and args.thinking_budget is not None:
                # mlx-vlm cannot enforce a thinking budget while drafting.
                args.thinking_budget = None
        return args

    # GLM-5-Next's DSA CacheList holds sub-caches that stay empty for very short
    # prompts; checkpointing them for the prefix cache must not crash.
    def batch_extract(self_, idx):
        if self_.keys is None:
            return cache_mod.KVCache()
        return _ORIG["batch_extract"](self_, idx)

    cache_mod.BatchKVCache.extract = batch_extract

    # mlx-vlm raises the wired limit to the whole Metal working set (~115 GB here)
    # for the life of every BatchGenerator; keep the engine's limit instead.
    @contextlib.contextmanager
    def wired_limit(model, streams=None):
        if _ACTIVE is None:
            with _ORIG["wired_limit"](model, streams):
                yield
        else:
            yield

    for mod in (ar, common, diffusion, dispatch, gen_pkg):
        mod.wired_limit = wired_limit
    srv.get_cached_model = get_cached_model
    apc._metal_working_set_bytes = working_set
    gen._check_configured_context_budget = check_context
    RG.generate = with_args(_ORIG["generate"])
    RG.validate_context_budget = with_args(_ORIG["validate"])
    RG._step = guarded_step
    rn._build_gen_args = build_gen_args


def detect_mtp(model: str) -> Optional[str]:
    """Drafter path when the checkpoint ships an MTP head: bundled tensors, or the
    `mtp/` drafter directory that build/ writes next to the main weights."""
    from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter

    path = Path(model).expanduser()
    if not path.is_dir():
        return None
    if (path / "mtp" / "config.json").exists():
        return str(path / "mtp")
    return model if detect_mtp_splitter(path) is not None else None


def resolve_draft_model(cfg: ServeConfig) -> Optional[str]:
    if cfg.mtp == "off":
        return None
    if cfg.draft_model:
        return cfg.draft_model
    drafter = detect_mtp(cfg.model)
    if drafter:
        return drafter
    if cfg.mtp == "on" and not Path(cfg.model).expanduser().is_dir():
        return cfg.model  # remote repo: mlx-vlm resolves its bundled head after download
    if cfg.mtp == "on":
        raise ValueError(f"--mtp on but {cfg.model} has no MTP head; pass --draft-model")
    return None


class Engine:
    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.load_lock = threading.RLock()
        self._count_lock = threading.Lock()
        self.inflight = 0
        self.requests_served = 0
        self.last_activity = time.monotonic()
        self.load_count = 0
        self.unload_count = 0
        self.last_load_s: Optional[float] = None
        self.last_unload_reason: Optional[str] = None
        self.limits: dict = {}
        self.draft_model = resolve_draft_model(cfg)
        self.mtp_active = False
        self.draft_kind: Optional[str] = None
        self.kv_quant_error: Optional[str] = None
        self.prefill_projection_tokens: Optional[int] = None
        self.guard = memory.MemoryGuard(cfg.memory_limit_bytes)
        self._stop = threading.Event()
        self._idle_thread: Optional[threading.Thread] = None
        self._cache_cleared = True
        self.srv = None
        self.app_mod = None

    # ---- mlx_vlm.server integration -------------------------------------

    def install(self):
        global _ACTIVE
        if _ACTIVE is not None and _ACTIVE is not self:
            _ACTIVE.shutdown()
        if self.cfg.weights_bytes is None:
            try:
                self.cfg.weights_bytes = memory.weights_bytes(self.cfg.model)
            except Exception as e:
                logger.warning("Could not size %s yet: %s", self.cfg.model, e)
        for name in CLEARED_ENV:
            os.environ.pop(name, None)
        os.environ.update(self.cfg.mlx_vlm_env(self.draft_model))
        _patch_mlx_vlm()
        self._prune_stale_prefix_caches()

        import mlx_vlm.server as srv
        from mlx_vlm.server.runtime_config import RuntimeConfig

        self.srv = srv
        self.app_mod = importlib.import_module("mlx_vlm.server.app")
        srv.runtime.config = RuntimeConfig.from_env()
        _ACTIVE = self

    def _prune_stale_prefix_caches(self):
        """Drop SSD prefix caches left by earlier builds of this served name."""
        if not (self.cfg.prefix_cache and self.cfg.prefix_cache_disk_gb > 0):
            return
        current = Path(self.cfg.prefix_cache_path())
        root = self.cfg.prefix_cache_root
        if not root.is_dir():
            return
        for d in root.glob(f"{self.cfg.name}-*"):
            stale = d.is_dir() and d != current and len(d.name) == len(current.name)
            if stale:
                logger.info("Removing stale prefix cache %s", d)
                shutil.rmtree(d, ignore_errors=True)

    # ---- load / unload ---------------------------------------------------

    def is_loaded(self) -> bool:
        if self.srv is None:
            return False
        cache = self.app_mod._model_cache_registry().for_kind("text_generation")
        return (
            bool(cache)
            and cache.get("model_path") == self.cfg.model
            and cache.get("response_generator") is not None
        )

    def get_cached_model(self, model_path=None, adapter_path=None, *, model_kind="auto"):
        from fastapi import HTTPException

        if model_kind not in ("auto", "text_generation"):
            raise HTTPException(
                status_code=400,
                detail=f"This server only serves text generation with {self.cfg.name}.",
            )
        try:
            return self.ensure_loaded()
        except memory.ModelTooLarge as e:
            raise HTTPException(status_code=507, detail=str(e)) from e

    def ensure_loaded(self):
        with self.load_lock:
            if self.is_loaded():
                return _ORIG["get_cached_model"](self.cfg.model, None, model_kind="text_generation")
            memory.check_fits(self.cfg)
            self.limits = memory.apply_limits(self.cfg)
            t0 = time.perf_counter()
            logger.info(
                "Loading %s (%.1f GB weights, cap %.1f GB, MTP drafter: %s)",
                self.cfg.model,
                (self.cfg.weights_bytes or 0) / GB,
                self.cfg.memory_limit_gb,
                self.draft_model or "none",
            )
            try:
                result = _ORIG["get_cached_model"](self.cfg.model, None, model_kind="text_generation")
            except Exception:
                if not self.draft_model or self.cfg.mtp == "on" or self.cfg.draft_model:
                    memory.release_wired()
                    raise
                logger.exception("Load with MTP drafter failed; retrying without speculative decoding.")
                self._disable_draft()
                result = _ORIG["get_cached_model"](self.cfg.model, None, model_kind="text_generation")
            rg = self.srv.runtime.response_generator
            self._bound_prefill_caches(rg.model)
            self._warmup(rg)
            self.mtp_active = getattr(rg, "draft_model", None) is not None
            self.draft_kind = getattr(rg, "draft_kind", None) if self.mtp_active else None
            self.last_load_s = time.perf_counter() - t0
            self.load_count += 1
            self.touch()
            logger.info(
                "Model ready in %.1fs: active %.2f GB, speculative=%s",
                self.last_load_s,
                mx.get_active_memory() / GB,
                self.draft_kind or "off",
            )
            return result

    def _bound_prefill_caches(self, model):
        """GLM-5-Next keeps per-head projected K/V (~0.7 MB/token at full size)
        through prefill to speed it up; cap it at a quarter of the KV reserve."""
        text = getattr(model.config, "text_config", None)
        if not str(getattr(text, "model_type", "")).startswith("glm5_next"):
            return
        from mlx_vlm.models.glm5_next import language as glm

        dsa_layers = sum(t != "linear_attention" for t in text.layer_types)
        per_token = dsa_layers * text.num_attention_heads * (text.qk_nope_head_dim + text.v_head_dim) * 2
        budget = int(self.cfg.kv_reserve_gb * GB) // 4
        _ORIG.setdefault("glm_projected", glm._MAX_PROJECTED_PREFILL_TOKENS)
        glm._MAX_PROJECTED_PREFILL_TOKENS = min(_ORIG["glm_projected"], budget // max(1, per_token))
        self.prefill_projection_tokens = glm._MAX_PROJECTED_PREFILL_TOKENS
        logger.info(
            "GLM projected prefill K/V capped at %d tokens (%.0f KB/token)",
            glm._MAX_PROJECTED_PREFILL_TOKENS,
            per_token / 1024,
        )

    def _probe(self, rg):
        from mlx_vlm.server.generation import GenerationArguments

        args = GenerationArguments(max_tokens=2, temperature=0.0)
        _, tokens = rg.generate("warm up", args=args)
        for _ in tokens:
            pass

    def _warmup(self, rg):
        """One tiny generation through the real server path: compiles kernels and
        verifies the KV-quantization request is supported by this architecture."""
        self.kv_quant_error = None
        try:
            self._probe(rg)
            return
        except Exception as e:
            if not rg.kv_bits:
                raise
            self.kv_quant_error = f"{type(e).__name__}: {e}"
        logger.warning(
            "KV cache quantization (%s bits) is not supported for this architecture in "
            "mlx-vlm's batch path; serving with full-precision KV. (%s)",
            rg.kv_bits,
            self.kv_quant_error,
        )
        rg.kv_bits = rg.kv_key_bits = rg.kv_value_bits = None
        rg.kv_key_scheme = rg.kv_value_scheme = None
        self._probe(rg)

    def _disable_draft(self):
        self.draft_model = None
        os.environ.pop("MLX_VLM_DRAFT_MODEL", None)
        self.srv.runtime.config.draft_model = None
        self.srv.runtime.config.draft_kind = None

    def unload(self, reason: str, *, force: bool = False) -> bool:
        with self.load_lock:
            if not self.is_loaded():
                return False
            if self.inflight and not force:
                return False
            logger.info("Unloading %s (%s)", self.cfg.model, reason)
            self.app_mod.unload_model_sync()
            self.mtp_active = False
            gc.collect()
            mx.clear_cache()
            memory.release_wired()
            self.limits["wired_limit"] = 0
            self.unload_count += 1
            self.last_unload_reason = reason
            return True

    # ---- activity / idle -------------------------------------------------

    def touch(self):
        self.last_activity = time.monotonic()
        self._cache_cleared = False

    def begin(self):
        with self._count_lock:
            self.inflight += 1
        self.touch()

    def end(self):
        with self._count_lock:
            self.inflight -= 1
            self.requests_served += 1
        self.touch()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity

    def _idle_tick(self):
        if self.inflight or not self.is_loaded():
            return
        idle = self.idle_seconds()
        if self.cfg.idle_unload_s and idle >= self.cfg.idle_unload_s:
            self.unload(f"idle {idle:.0f}s")
        elif idle >= 10 and not self._cache_cleared:
            mx.clear_cache()
            self._cache_cleared = True

    def start(self):
        if self._idle_thread is not None:
            return
        interval = max(0.5, min(5.0, (self.cfg.idle_unload_s or 20) / 4))

        def loop():
            while not self._stop.wait(interval):
                try:
                    self._idle_tick()
                except Exception:
                    logger.exception("Idle monitor error")

        self._idle_thread = threading.Thread(target=loop, name="serve-idle", daemon=True)
        self._idle_thread.start()

    def shutdown(self):
        self._stop.set()
        if self._idle_thread is not None:
            self._idle_thread.join(timeout=5)
            self._idle_thread = None
        self.unload("shutdown", force=True)

    # ---- reporting -------------------------------------------------------

    def status(self) -> dict:
        rt = self.srv.runtime if self.srv else None
        rg = rt.response_generator if rt else None
        queue_depth = rg.requests.qsize() if rg is not None else 0
        apc = rt.apc_manager.stats_snapshot() if rt and rt.apc_manager else None
        return {
            "model": self.cfg.model,
            "served_name": self.cfg.name,
            "loaded": self.is_loaded(),
            "weights_gb": round((self.cfg.weights_bytes or 0) / GB, 3),
            "speculative": {
                "drafter": self.draft_model,
                "active": self.mtp_active,
                "kind": self.draft_kind,
            },
            "kv_quant": {
                "requested_bits": self.cfg.kv_bits,
                "active_bits": getattr(rg, "kv_bits", None),
                "error": self.kv_quant_error,
            },
            "glm_projected_prefill_tokens": self.prefill_projection_tokens,
            "prefix_cache_dir": self.cfg.prefix_cache_path() if self.cfg.prefix_cache else None,
            "inflight": self.inflight,
            "queue_depth": queue_depth,
            "requests_served": self.requests_served,
            "idle_s": round(self.idle_seconds(), 1),
            "idle_unload_s": self.cfg.idle_unload_s,
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "last_load_s": self.last_load_s,
            "last_unload_reason": self.last_unload_reason,
            "memory": {
                **memory.snapshot(),
                "cap_gb": self.cfg.memory_limit_gb,
                "wired_limit_gb": round(self.limits.get("wired_limit", 0) / GB, 3),
                "guard_trips": self.guard.trips,
            },
            "prefix_cache": apc,
            "config": self.cfg.to_dict(),
        }
