"""uv run python -m serve --model models/<name> [options]"""

import argparse
import json
import logging
import os
import sys

from .config import ServeConfig


def _env(name, default, cast=str):
    raw = os.environ.get(f"SERVE_{name}")
    return default if raw in (None, "") else cast(raw)


def _bool(raw: str) -> bool:
    return str(raw).lower() in ("1", "true", "yes", "on")


def parse_args(argv=None) -> ServeConfig:
    d = ServeConfig(model="")
    p = argparse.ArgumentParser(
        prog="python -m serve",
        description="Memory-capped OpenAI + Anthropic compatible MLX server (one pinned model).",
    )
    p.add_argument("--model", default=_env("MODEL", None), required=_env("MODEL", None) is None,
                   help="Local model dir (or HF repo id) loadable by mlx_vlm.load.")
    p.add_argument("--served-name", default=_env("SERVED_NAME", None),
                   help="Name reported by /v1/models (any requested model name is accepted).")
    p.add_argument("--host", default=_env("HOST", d.host))
    p.add_argument("--port", type=int, default=_env("PORT", d.port, int))
    p.add_argument("--api-key", default=_env("API_KEY", None),
                   help="Require 'Authorization: Bearer <key>' on every endpoint.")

    m = p.add_argument_group("memory")
    m.add_argument("--memory-limit-gb", type=float, default=_env("MEMORY_LIMIT_GB", d.memory_limit_gb, float),
                   help="Hard cap for the whole server process (default: %(default)s).")
    m.add_argument("--kv-reserve-gb", type=float, default=_env("KV_RESERVE_GB", d.kv_reserve_gb, float),
                   help="Headroom required above the weights for KV/state/activations; "
                        "a model with weights + reserve > cap is refused (default: %(default)s).")
    m.add_argument("--wired-limit-gb", type=float, default=_env("WIRED_LIMIT_GB", None, float),
                   help="mx.set_wired_limit (default: weights + reserve, within the cap).")
    m.add_argument("--cache-limit-gb", type=float, default=_env("CACHE_LIMIT_GB", d.cache_limit_gb, float),
                   help="MLX free-buffer cache limit (default: %(default)s).")
    m.add_argument("--idle-unload-s", type=float, default=_env("IDLE_UNLOAD_S", d.idle_unload_s, float),
                   help="Unload the model after this many idle seconds; 0 disables (default: %(default)s).")
    m.add_argument("--no-preload", action="store_true", default=_env("NO_PRELOAD", False, _bool),
                   help="Load on the first request instead of at startup.")

    g = p.add_argument_group("generation")
    g.add_argument("--max-context", type=int, default=_env("MAX_CONTEXT", d.max_context, int),
                   help="Prompt + generated token limit (default: %(default)s).")
    g.add_argument("--max-output-tokens", type=int, default=_env("MAX_OUTPUT_TOKENS", d.max_output_tokens, int),
                   help="Upper bound applied to every request's max_tokens (default: %(default)s).")
    g.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                   default=_env("ENABLE_THINKING", True, _bool),
                   help="Requests that do not choose get thinking off (template-permitting) instead of "
                        "reasoning returned as thinking blocks / reasoning_content.")
    g.add_argument("--prefill-step-size", type=int, default=_env("PREFILL_STEP_SIZE", None, int))
    g.add_argument("--kv-bits", type=float, default=_env("KV_BITS", None, float),
                   help="Quantize the attention KV cache (e.g. 8, 4; 3.5 with turboquant).")
    g.add_argument("--kv-quant-scheme", choices=("uniform", "turboquant"),
                   default=_env("KV_QUANT_SCHEME", d.kv_quant_scheme))
    g.add_argument("--kv-group-size", type=int, default=_env("KV_GROUP_SIZE", d.kv_group_size, int))
    g.add_argument("--quantized-kv-start", type=int, default=_env("QUANTIZED_KV_START", d.quantized_kv_start, int),
                   help="Token position after which the KV cache is quantized.")
    g.add_argument("--mtp", choices=("auto", "on", "off"), default=_env("MTP", d.mtp),
                   help="MTP speculative decoding with the checkpoint's own head (default: %(default)s).")
    g.add_argument("--draft-model", default=_env("DRAFT_MODEL", None),
                   help="Separate drafter checkpoint (otherwise the model's bundled MTP head).")
    g.add_argument("--draft-block-size", type=int, default=_env("DRAFT_BLOCK_SIZE", None, int))

    c = p.add_argument_group("prefix cache")
    c.add_argument("--no-prefix-cache", action="store_true", default=_env("NO_PREFIX_CACHE", False, _bool))
    c.add_argument("--prefix-cache-memory-gb", type=float, default=_env("PREFIX_CACHE_MEMORY_GB", None, float),
                   help="Resident prefix-cache budget (default: what the cap leaves, 0.5-16 GB).")
    c.add_argument("--prefix-cache-disk-gb", type=float,
                   default=_env("PREFIX_CACHE_DISK_GB", d.prefix_cache_disk_gb, float),
                   help="SSD tier size, survives idle unloads and restarts; 0 disables (default: %(default)s).")
    c.add_argument("--prefix-cache-entries", type=int,
                   default=_env("PREFIX_CACHE_ENTRIES", d.prefix_cache_entries, int),
                   help="Resident whole-prefix checkpoints kept for hybrid models (default: %(default)s).")
    c.add_argument("--prefix-cache-dir", default=_env("PREFIX_CACHE_DIR", None),
                   help="SSD tier root; one subdir per checkpoint fingerprint (default: ~/.cache/local-ai/prefix-cache).")

    p.add_argument("--dry-run", action="store_true",
                   help="Print the memory-fit and MTP decision as JSON, then exit (2 if refused).")
    p.add_argument("--log-level", default=_env("LOG_LEVEL", d.log_level),
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    a = p.parse_args(argv)
    cfg = ServeConfig(
        model=a.model,
        served_name=a.served_name,
        host=a.host,
        port=a.port,
        api_key=a.api_key,
        memory_limit_gb=a.memory_limit_gb,
        kv_reserve_gb=a.kv_reserve_gb,
        wired_limit_gb=a.wired_limit_gb,
        cache_limit_gb=a.cache_limit_gb,
        max_context=a.max_context,
        max_output_tokens=a.max_output_tokens,
        enable_thinking=a.enable_thinking,
        kv_bits=a.kv_bits,
        kv_quant_scheme=a.kv_quant_scheme,
        kv_group_size=a.kv_group_size,
        quantized_kv_start=a.quantized_kv_start,
        mtp=a.mtp,
        draft_model=a.draft_model,
        draft_block_size=a.draft_block_size,
        prefix_cache=not a.no_prefix_cache,
        prefix_cache_memory_gb=a.prefix_cache_memory_gb,
        prefix_cache_disk_gb=a.prefix_cache_disk_gb,
        prefix_cache_dir=a.prefix_cache_dir,
        prefix_cache_entries=a.prefix_cache_entries,
        idle_unload_s=a.idle_unload_s,
        preload=not a.no_preload,
        prefill_step_size=a.prefill_step_size,
        log_level=a.log_level,
    )
    cfg.dry_run = a.dry_run
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("serve")

    from . import memory

    error = None
    try:
        memory.check_fits(cfg)
    except (memory.ModelTooLarge, FileNotFoundError) as e:
        error = str(e)
    if getattr(cfg, "dry_run", False):
        from .engine import detect_mtp

        print(json.dumps({
            "model": cfg.model,
            "fits": error is None,
            "error": error,
            "weights_gb": round((cfg.weights_bytes or memory.weights_bytes(cfg.model)) / (1 << 30), 3)
            if error is None or "Refusing" in error else None,
            "memory_limit_gb": cfg.memory_limit_gb,
            "kv_reserve_gb": cfg.kv_reserve_gb,
            "mtp_head": detect_mtp(cfg.model),
            "prefix_cache_memory_gb": round(cfg.prefix_cache_memory_bytes() / (1 << 30), 3),
        }, indent=2))
        sys.exit(0 if error is None else 2)
    if error:
        log.error("%s", error)
        sys.exit(2)

    import uvicorn

    from .app import create_app

    app = create_app(cfg)
    log.info(
        "Serving %s as %r on http://%s:%d (cap %.0f GB, weights %.1f GB, MTP %s, prefix cache %s)",
        cfg.model, cfg.name, cfg.host, cfg.port, cfg.memory_limit_gb,
        (cfg.weights_bytes or 0) / (1 << 30), app.state.engine.draft_model or "off",
        "on" if cfg.prefix_cache else "off",
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1, server_header=False,
                log_level=cfg.log_level.lower(), timeout_keep_alive=75)


if __name__ == "__main__":
    main()
