"""Stock mlx-vlm server with co-tenant memory limits (see docs/research/mac.md).

mlx-vlm wires up to max_recommended_working_set_size (~107 GiB) on every generate; we clamp
every set_wired_limit call to WIRE_GB (0 = no wiring) and set a soft memory/cache limit.
"""

import os

import mlx.core as mx

GB = 1 << 30
CAP = float(os.environ.get("CAP_GB", "80")) * GB
WIRE = int(float(os.environ.get("WIRE_GB", "0")) * GB)

_set_wired = mx.set_wired_limit
mx.set_wired_limit = lambda n: _set_wired(min(n, WIRE))
mx.set_memory_limit(int(CAP - 4 * GB))
mx.set_cache_limit(int(float(os.environ.get("CACHE_GB", "2")) * GB))

import secrets  # noqa: E402

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
        return await call_next(request)

main()
