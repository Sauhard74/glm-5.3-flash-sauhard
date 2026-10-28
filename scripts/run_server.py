"""Stock mlx-vlm server with co-tenant memory limits (see docs/research/mac.md).

mlx-vlm wires up to max_recommended_working_set_size (~107 GiB) on every generate; we clamp
every set_wired_limit call to WIRE_GB (0 = no wiring) and set a soft memory/cache limit.
"""

import logging
import os
import sys

import mlx.core as mx

GB = 1 << 30
CAP = float(os.environ.get("CAP_GB", "80")) * GB
WIRE = int(float(os.environ.get("WIRE_GB", "0")) * GB)

_set_wired = mx.set_wired_limit
mx.set_wired_limit = lambda n: _set_wired(min(n, WIRE))
mx.set_memory_limit(int(CAP - 4 * GB))
mx.set_cache_limit(int(float(os.environ.get("CACHE_GB", "2")) * GB))

import secrets  # noqa: E402

# GLM keeps a per-head projected KV copy during prefill (22-24 GiB extra at >= 32K context).
# 0 disables it; verified bit-exact in docs/research/decode.md (Rec 1, MASTER_PLAN R13).
import mlx_vlm.models.glm5_next.language as _glm_lang  # noqa: E402

_glm_lang._MAX_PROJECTED_PREFILL_TOKENS = int(os.environ.get("GLM_MAX_PROJECTED_PREFILL", "0"))

# With it disabled the projected cache stays empty (keys None), which BatchKVCache.extract cannot slice when the
# server splits a finished request out of the batch; an empty cache is what it holds (merge() accepts those).
from mlx_vlm.models import cache as _cache  # noqa: E402

_extract = _cache.BatchKVCache.extract
_cache.BatchKVCache.extract = lambda self, idx: _cache.KVCache() if self.keys is None else _extract(self, idx)

# Shared-prefix checkpoints, weights-keyed disk namespace, whole-root disk cap (see prompt_cache.py).
import prompt_cache  # noqa: E402,F401

# One model per server. Stock swaps models on any request naming another one, and the old one stays resident while
# its in-flight request drains: a client asking for a different name loaded a second 72 GB copy (142 GB, deep swap).
# Every text-generation request is served by the model the server was started with, whatever name it sends.
import importlib  # noqa: E402

_app = importlib.import_module("mlx_vlm.server.app")  # the module; the package's `app` attribute is the FastAPI app
_argv = sys.argv[1:]
PINNED = _argv[_argv.index("--model") + 1] if "--model" in _argv else None
_get_cached_model = _app.get_cached_model


def _pinned_get_cached_model(model_path, *args, model_kind="auto", **kwargs):
    if PINNED and model_kind in ("auto", "text_generation") and model_path != PINNED:
        logging.getLogger("mlx_vlm").info("Request named %s; serving the pinned model %s", model_path, PINNED)
        model_path = PINNED
    return _get_cached_model(model_path, *args, model_kind=model_kind, **kwargs)


if PINNED:  # routes resolve it at call time, package attribute first (app._server_package_attr)
    import mlx_vlm.server as _server  # noqa: E402

    _app.get_cached_model = _server.get_cached_model = _pinned_get_cached_model

from mlx_vlm.server import app as server_app  # noqa: E402
from mlx_vlm.server.cli import main  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

# Stock --api-key only guards management routes; require the key on every route, OpenAI (Bearer) or Anthropic (x-api-key) style.
API_KEY = os.environ.get("MLX_VLM_SERVER_API_KEY")
if API_KEY:
    @server_app.middleware("http")
    async def require_api_key(request, call_next):
        supplied = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, API_KEY):
            return JSONResponse({"error": {"type": "authentication_error", "message": "invalid api key"}}, status_code=401)
        # mlx-vlm's own management check only reads Authorization; mirror a valid x-api-key into it.
        if not request.headers.get("authorization"):
            request.scope["headers"] = [*request.scope["headers"], (b"authorization", f"Bearer {API_KEY}".encode())]
        return await call_next(request)

main()
