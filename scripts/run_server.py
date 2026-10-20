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

from mlx_vlm.server.cli import main  # noqa: E402

main()
