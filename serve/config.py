import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

GB = 1 << 30


@dataclass
class ServeConfig:
    model: str
    served_name: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8642
    api_key: Optional[str] = None

    # Memory: the whole process is held to memory_limit_gb.
    memory_limit_gb: float = 80.0
    kv_reserve_gb: float = 8.0
    wired_limit_gb: Optional[float] = None  # None: weights + kv_reserve, capped
    cache_limit_gb: float = 4.0

    max_context: int = 131072
    max_output_tokens: int = 32768
    enable_thinking: bool = True  # both candidates' templates think by default

    kv_bits: Optional[float] = None
    kv_quant_scheme: str = "uniform"
    kv_group_size: int = 64
    quantized_kv_start: int = 0

    mtp: str = "auto"  # auto | on | off
    draft_model: Optional[str] = None
    draft_block_size: Optional[int] = None

    prefix_cache: bool = True
    prefix_cache_memory_gb: Optional[float] = None  # None: what the cap leaves
    prefix_cache_disk_gb: float = 32.0  # 0 disables the SSD tier
    prefix_cache_dir: Optional[str] = None
    prefix_cache_entries: int = 4  # resident whole-prefix checkpoints (hybrid models)

    idle_unload_s: float = 900.0  # 0 disables
    preload: bool = True
    token_timeout_s: float = 1800.0
    prefill_step_size: Optional[int] = None
    log_level: str = "INFO"

    weights_bytes: Optional[int] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.served_name or Path(self.model.rstrip("/")).name

    @property
    def memory_limit_bytes(self) -> int:
        return int(self.memory_limit_gb * GB)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("api_key")
        out["api_key_set"] = bool(self.api_key)
        return out

    def prefix_cache_memory_bytes(self) -> int:
        if self.prefix_cache_memory_gb is not None:
            return int(self.prefix_cache_memory_gb * GB)
        spare = self.memory_limit_bytes - (self.weights_bytes or 0) - int(self.kv_reserve_gb * GB)
        return max(GB // 2, min(16 * GB, spare))

    @property
    def prefix_cache_root(self) -> Path:
        return Path(os.path.expanduser(self.prefix_cache_dir or "~/.cache/local-ai/prefix-cache"))

    def prefix_cache_path(self) -> str:
        """SSD tier directory keyed by the checkpoint's content, so a rebuilt model
        at the same path never restores stale KV/state."""
        return str(self.prefix_cache_root / f"{self.name}-{model_fingerprint(self.model)}")

    def mlx_vlm_env(self, draft_model: Optional[str]) -> dict:
        """Environment consumed by mlx_vlm.server (it reads most knobs from env)."""
        env = {
            "MLX_VLM_MAX_NUM_SEQS": "1",
            "MAX_KV_SIZE": str(self.max_context),
            "MLX_VLM_MAX_TOKENS": str(self.max_output_tokens),
            "MLX_VLM_ENABLE_THINKING": "1" if self.enable_thinking else "0",
            "MLX_VLM_TOKEN_QUEUE_TIMEOUT": str(self.token_timeout_s),
            "KV_QUANT_SCHEME": self.kv_quant_scheme,
            "KV_GROUP_SIZE": str(self.kv_group_size),
            "QUANTIZED_KV_START": str(self.quantized_kv_start),
            "APC_ENABLED": "1" if self.prefix_cache else "0",
            "APC_MEMORY_MAX_GB": f"{self.prefix_cache_memory_bytes() / GB:.3f}",
            "APC_DISK_ENABLED": "1" if self.prefix_cache_disk_gb > 0 else "0",
            "APC_DISK_MAX_GB": str(self.prefix_cache_disk_gb),
            "APC_CHECKPOINT_ENTRIES": str(self.prefix_cache_entries),
        }
        if self.prefix_cache and self.prefix_cache_disk_gb > 0:
            env["APC_DISK_PATH"] = self.prefix_cache_path()
        if self.kv_bits:
            env["KV_BITS"] = str(self.kv_bits)
        if draft_model:
            env["MLX_VLM_DRAFT_MODEL"] = draft_model
        if self.draft_block_size:
            env["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(self.draft_block_size)
        if self.prefill_step_size:
            env["PREFILL_STEP_SIZE"] = str(self.prefill_step_size)
        if self.api_key:
            env["MLX_VLM_SERVER_API_KEY"] = self.api_key
        return env


def model_fingerprint(model: str) -> str:
    h = hashlib.sha256()
    path = Path(model).expanduser()
    if not path.is_dir():
        h.update(model.encode())
        return h.hexdigest()[:12]
    for f in sorted(path.iterdir()):
        if f.suffix in (".json", ".safetensors", ".jinja"):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    config = path / "config.json"
    if config.exists():
        h.update(config.read_bytes())
    return h.hexdigest()[:12]


CLEARED_ENV = (
    "KV_BITS",
    "MLX_VLM_DRAFT_MODEL",
    "MLX_VLM_DRAFT_KIND",
    "MLX_VLM_DRAFT_BLOCK_SIZE",
    "MLX_VLM_PRELOAD_MODEL",
    "APC_DISK_PATH",
    "APC_CHECKPOINT_ENTRIES",
    "MLX_VLM_SERVER_API_KEY",
)
