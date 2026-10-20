"""Streaming reader for a HF safetensors snapshot.

Tensors are read one at a time straight from the file byte ranges (no mmap of whole
shards), optionally only a subset of rows along axis 0 (expert slicing), and block-FP8
weights are dequantized to bf16 on read.
"""

import json
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from mlx_vlm.fp8 import _dequantize_fp8_weight

_NP_DTYPES = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, mx.float16),
    "F32": (np.float32, mx.float32),
    "F64": (np.float64, mx.float32),
    "I64": (np.int64, mx.int64),
    "I32": (np.int32, mx.int32),
    "I16": (np.int16, mx.int16),
    "I8": (np.int8, mx.int8),
    "U64": (np.uint64, mx.uint64),
    "U32": (np.uint32, mx.uint32),
    "U16": (np.uint16, mx.uint16),
    "U8": (np.uint8, mx.uint8),
    "BOOL": (np.bool_, mx.bool_),
    "F8_E4M3": (np.uint8, mx.uint8),
    "F8_E4M3FN": (np.uint8, mx.uint8),
}
_ITEMSIZE = {k: np.dtype(v[0]).itemsize for k, v in _NP_DTYPES.items()}
FP8_DTYPES = ("F8_E4M3", "F8_E4M3FN")
SCALE_SUFFIX = "_scale_inv"


def read_header(path: Path) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class TensorInfo:
    __slots__ = ("key", "file", "dtype", "shape", "start", "end")

    def __init__(self, key, file, dtype, shape, start, end):
        self.key, self.file, self.dtype = key, file, dtype
        self.shape, self.start, self.end = tuple(shape), start, end

    @property
    def is_fp8(self) -> bool:
        return self.dtype in FP8_DTYPES

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class Source:
    """A HF snapshot directory (config.json + *.safetensors [+ index])."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path / "config.json") as f:
            self.config = json.load(f)
        files = sorted(
            p
            for p in self.path.glob("*.safetensors")
            if not p.name.endswith("consolidated.safetensors")
        )
        if not files:
            raise FileNotFoundError(f"no safetensors in {self.path}")
        self.tensors: Dict[str, TensorInfo] = {}
        for file in files:
            header, data_start = read_header(file)
            for key, meta in header.items():
                begin, end = meta["data_offsets"]
                self.tensors[key] = TensorInfo(
                    key,
                    file,
                    meta["dtype"],
                    meta["shape"],
                    data_start + begin,
                    data_start + end,
                )
        self._handles = {}

    def keys(self) -> List[str]:
        return [k for k in self.tensors if not k.endswith(SCALE_SUFFIX)]

    def __contains__(self, key):
        return key in self.tensors

    def info(self, key) -> TensorInfo:
        return self.tensors[key]

    def logical_dtype(self, key):
        info = self.tensors[key]
        if info.is_fp8 and key + SCALE_SUFFIX in self.tensors:
            return mx.bfloat16
        return _NP_DTYPES[info.dtype][1]

    def _handle(self, file):
        if file not in self._handles:
            self._handles[file] = open(file, "rb")
        return self._handles[file]

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles = {}

    def _read_raw(self, info: TensorInfo, rows: Optional[Sequence[int]] = None):
        np_dtype, mx_dtype = _NP_DTYPES[info.dtype]
        f = self._handle(info.file)
        shape = list(info.shape)
        if rows is None:
            spans = [(info.start, info.nbytes)]
        else:
            if list(rows) != sorted(set(rows)):
                raise ValueError(f"rows for {info.key} must be sorted and unique")
            row_bytes = info.nbytes // shape[0] if shape[0] else 0
            spans = [(info.start + b * row_bytes, (e - b) * row_bytes) for b, e in _runs(rows)]
            shape[0] = len(rows)
        buf = np.empty(sum(n for _, n in spans), dtype=np.uint8)
        view, pos = memoryview(buf), 0
        for start, n in spans:
            f.seek(start)
            if f.readinto(view[pos : pos + n]) != n:
                raise IOError(f"short read for {info.key} in {info.file}")
            pos += n
        arr = buf.view(np_dtype).reshape(shape)
        if info.dtype == "F64":
            arr = arr.astype(np.float32)
        out = mx.array(arr)
        if info.dtype == "BF16":
            out = out.view(mx.bfloat16)
        return out

    def read(self, key: str, rows: Optional[Sequence[int]] = None) -> mx.array:
        """Read a tensor (dequantizing block-FP8). ``rows`` selects along axis 0."""
        info = self.tensors[key]
        value = self._read_raw(info, rows)
        scale_key = key + SCALE_SUFFIX
        if info.is_fp8 and scale_key in self.tensors:
            scale = self._read_raw(self.tensors[scale_key])
            if value.ndim == 2:
                value = _dequantize_fp8_weight(value, scale)
            elif value.ndim == 3:
                if rows is not None:
                    scale = scale[mx.array(list(rows))]
                value = mx.stack(
                    [_dequantize_fp8_weight(value[i], scale[i]) for i in range(len(value))]
                )
            else:
                raise ValueError(f"unsupported FP8 tensor rank for {key}: {value.shape}")
            value = value.astype(mx.bfloat16)  # fp32 scale_inv would otherwise promote to fp32
        return value

    def placeholder(self, key: str, rows: Optional[Sequence[int]] = None) -> mx.array:
        """Lazy zeros with the logical shape/dtype of ``read`` (never evaluated)."""
        shape = list(self.tensors[key].shape)
        if rows is not None:
            shape[0] = len(rows)
        return mx.zeros(shape, dtype=self.logical_dtype(key))

    def total_source_bytes(self, keys: Iterable[str]) -> int:
        total = 0
        for key in keys:
            total += self.tensors[key].nbytes
            if key + SCALE_SUFFIX in self.tensors:
                total += self.tensors[key + SCALE_SUFFIX].nbytes
        return total


def _runs(rows: Sequence[int]):
    rows = list(rows)
    if not rows:
        return
    begin = prev = rows[0]
    for r in rows[1:]:
        if r != prev + 1:
            yield begin, prev + 1
            begin = r
        prev = r
    yield begin, prev + 1
