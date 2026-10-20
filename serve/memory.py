import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .config import GB, ServeConfig

logger = logging.getLogger("serve.memory")


class ModelTooLarge(RuntimeError):
    status_code = 507


class MemoryCapExceeded(RuntimeError):
    pass


def _local_weight_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        files = [model_dir / s for s in sorted(shards)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{model_dir}: missing shards {missing[:3]}")
        return files
    return [
        f
        for f in sorted(model_dir.glob("*.safetensors"))
        if f.name != "consolidated.safetensors"
    ]


def weights_bytes(model: str) -> int:
    """Bytes of safetensors weights the model would make resident (MTP included)."""
    path = Path(model).expanduser()
    if path.is_dir():
        files = _local_weight_files(path)
        if not files:
            raise FileNotFoundError(f"No safetensors in {path}")
        return sum(f.stat().st_size for f in files)
    from huggingface_hub import HfApi

    info = HfApi().model_info(model, files_metadata=True)
    sizes = [
        s.size or 0
        for s in info.siblings or []
        if s.rfilename.endswith(".safetensors")
        and not s.rfilename.endswith("consolidated.safetensors")
    ]
    if not sizes:
        raise FileNotFoundError(f"No safetensors listed for {model}")
    return sum(sizes)


def check_fits(cfg: ServeConfig, nbytes: Optional[int] = None) -> int:
    nbytes = weights_bytes(cfg.model) if nbytes is None else nbytes
    need = nbytes + int(cfg.kv_reserve_gb * GB)
    if need > cfg.memory_limit_bytes:
        raise ModelTooLarge(
            f"Refusing to load {cfg.model}: weights {nbytes / GB:.1f} GB + "
            f"KV/state reserve {cfg.kv_reserve_gb:.1f} GB = {need / GB:.1f} GB exceeds "
            f"the {cfg.memory_limit_gb:.1f} GB memory cap."
        )
    cfg.weights_bytes = nbytes
    return nbytes


def wired_limit_bytes(cfg: ServeConfig) -> int:
    if cfg.wired_limit_gb is not None:
        want = int(cfg.wired_limit_gb * GB)
    else:
        want = (cfg.weights_bytes or 0) + int(cfg.kv_reserve_gb * GB)
    ceiling = int(mx.device_info().get("max_recommended_working_set_size", want))
    return max(0, min(want, cfg.memory_limit_bytes, ceiling))


def apply_limits(cfg: ServeConfig) -> dict:
    mx.set_memory_limit(cfg.memory_limit_bytes)
    mx.set_cache_limit(min(int(cfg.cache_limit_gb * GB), cfg.memory_limit_bytes))
    wired = wired_limit_bytes(cfg)
    mx.set_wired_limit(wired)
    return {
        "memory_limit": cfg.memory_limit_bytes,
        "cache_limit": int(cfg.cache_limit_gb * GB),
        "wired_limit": wired,
    }


def release_wired() -> None:
    mx.set_wired_limit(0)


def snapshot() -> dict:
    return {
        "active_gb": round(mx.get_active_memory() / GB, 3),
        "cache_gb": round(mx.get_cache_memory() / GB, 3),
        "peak_gb": round(mx.get_peak_memory() / GB, 3),
        "limit_gb": round(mx.get_memory_limit() / GB, 3),
    }


class MemoryGuard:
    """Checked between generation steps; frees what it can before failing requests."""

    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        self.trips = 0

    def over(self) -> bool:
        return mx.get_active_memory() > self.limit_bytes

    def enforce(self, apc_manager=None) -> None:
        if mx.get_active_memory() + mx.get_cache_memory() <= self.limit_bytes:
            return
        mx.clear_cache()
        if not self.over():
            return
        if apc_manager is not None:
            apc_manager._make_room()
            mx.clear_cache()
            if not self.over():
                logger.warning("Memory cap reached: dropped resident prefix cache.")
                return
        self.trips += 1
        raise MemoryCapExceeded(
            f"Memory cap exceeded: {mx.get_active_memory() / GB:.1f} GB active > "
            f"{self.limit_bytes / GB:.1f} GB. Shorten the context or raise --memory-limit-gb."
        )
