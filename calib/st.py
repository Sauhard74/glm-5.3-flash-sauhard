"""Read-only access to a sharded HF safetensors snapshot.

Tensors are fetched one at a time with lazy `mx.load` (nothing is resident until
evaluated); huge embedding tables are row-gathered through `np.memmap` so only the
touched pages are read. Source files are never written.
"""

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vlm.fp8 import _dequantize_fp8_weight

_NP_DTYPES = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
    "F8_E4M3": np.uint8,
}


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class Snapshot:
    def __init__(self, root):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            files = sorted(self.root.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors in {self.root}")
            self.weight_map = {k: f.name for f in files for k in read_header(f)[0]}
        self._headers = {}
        self._memmaps = {}

    def keys(self, prefix: str = "") -> list[str]:
        return [k for k in self.weight_map if k.startswith(prefix)]

    def __contains__(self, key):
        return key in self.weight_map

    def header(self, key: str) -> dict:
        fname = self.weight_map[key]
        if fname not in self._headers:
            self._headers[fname] = read_header(self.root / fname)
        return self._headers[fname][0][key]

    def shape(self, key: str) -> tuple:
        return tuple(self.header(key)["shape"])

    def dtype(self, key: str) -> str:
        return self.header(key)["dtype"]

    def load(self, keys) -> dict[str, mx.array]:
        """Lazy arrays for `keys` (grouped per shard file). FP8 comes back as uint8."""
        by_file = {}
        for k in keys:
            by_file.setdefault(self.weight_map[k], []).append(k)
        out = {}
        for fname, ks in by_file.items():
            arrays = mx.load(str(self.root / fname))
            for k in ks:
                out[k] = arrays[k]
            del arrays
        return out

    def load_dequantized(self, keys, dtype=mx.bfloat16) -> dict[str, mx.array]:
        """Like `load`, but fuses `X.weight` + `X.weight_scale_inv` (FP8 e4m3, block
        scaled) into one dequantized `X.weight`. Each tensor is evaluated before the
        next is read so at most one FP8 source is resident at a time."""
        keys = list(keys)
        scales = {k[: -len("_scale_inv")] for k in keys if k.endswith(".weight_scale_inv")}
        arrays = self.load(keys)
        out = {}
        for k in keys:
            if k.endswith(".weight_scale_inv"):
                continue
            v = arrays.pop(k)
            if k in scales:
                s = arrays.pop(k + "_scale_inv")
                v = _dequantize_fp8_weight(v, s.astype(mx.float32)).astype(dtype)
                mx.eval(v)
            elif self.dtype(k) == "F8_E4M3":
                raise ValueError(f"FP8 tensor without weight_scale_inv: {k}")
            out[k] = v
        return out

    def memmap(self, key: str) -> np.memmap:
        if key not in self._memmaps:
            h = self.header(key)
            fname = self.weight_map[key]
            start = self._headers[fname][1] + h["data_offsets"][0]
            self._memmaps[key] = np.memmap(
                self.root / fname, dtype=_NP_DTYPES[h["dtype"]], mode="r", offset=start,
                shape=tuple(h["shape"]),
            )
        return self._memmaps[key]

    def gather_rows(self, key: str, rows: np.ndarray) -> mx.array:
        mm = self.memmap(key)
        data = np.ascontiguousarray(mm[rows])
        arr = mx.array(data)
        if self.dtype(key) == "BF16":
            arr = arr.view(mx.bfloat16)
        return arr
