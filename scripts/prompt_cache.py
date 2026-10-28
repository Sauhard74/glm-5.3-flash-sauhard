"""Prompt-cache fixes on top of mlx-vlm's APC (imported by run_server.py before the server starts).

mlx-vlm 0.7.9 already reuses prefixes across requests and writes its checkpoints to an SSD tier
(~/.cache/mlx-vlm/apc, atomic tmp+rename, mtime LRU, APC_DISK_MAX_GB per model). For hybrid caches
(GLM's ArraysCache + MLA/indexer CacheList) a checkpoint restores only if the new prompt contains all of
its tokens, and each request snapshots just its end and the last 2048-multiple before it. A new Claude Code
session shares tools + system with the last one but diverges before either point, so it re-prefilled them.

- Shared-prefix checkpoint: if a prompt diverges from a recent one after >= APC_SHARED_PREFIX tokens
  (default 2048, 0 = off) and the lookup restored less than that, also snapshot at the divergence. It goes
  to disk like any checkpoint, so later sessions and restarts start there. Idea from Locals' `system`
  checkpoint (MIT, docs/prompt-cache); found here by token LCP so it needs no template knowledge.
- The disk namespace folds in a weights fingerprint. Stock keys it by the model path string, and
  models/* are symlinks repointed between quants: the old snapshots would restore into new weights.
- APC_DISK_TOTAL_GB (default 80, 0 = off) bounds the whole APC root. On load, other models' namespaces are
  pruned oldest-first to TOTAL - APC_DISK_MAX_GB; mlx-vlm keeps this model's own under APC_DISK_MAX_GB.
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
from mlx_vlm import apc as _apc
from mlx_vlm.apc_coordinator import APCCoordinator

GB = 1 << 30
SHARED_MIN = int(os.environ.get("APC_SHARED_PREFIX", "2048"))
DISK_TOTAL = int(float(os.environ.get("APC_DISK_TOTAL_GB", "80")) * GB)
BACKOFF = 16  # stop short of the divergence: the last shared token can merge (BPE) with what follows
CONTINUES = 64  # an LCP this close to an old prompt's end means the same conversation, covered by its end checkpoint

log = logging.getLogger("mlx_vlm.apc")
_lock = threading.Lock()
_recent = deque(maxlen=16)  # token ids of recent prompts (int32, <= 8 MB at 128K context)
_restored = OrderedDict()  # prompt key -> prefix length its lookup restored


def _key(ids):
    return hash(np.asarray(ids, dtype=np.int32).tobytes())


def _lcp(a, b):
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


_lookup = APCCoordinator.lookup


def lookup(self, token_ids, **kwargs):
    hit = _lookup(self, token_ids, **kwargs)
    with _lock:
        _restored[_key(token_ids)] = hit["prefix_len"] if hit else 0
        while len(_restored) > 16:
            _restored.popitem(last=False)
    return hit


_lengths = APCCoordinator.checkpoint_lengths


def checkpoint_lengths(self, token_ids, media_token_ids):
    lengths = _lengths(self, token_ids, media_token_ids)
    if SHARED_MIN <= 0 or not lengths:
        return lengths
    ids = np.asarray(token_ids, dtype=np.int32)
    with _lock:
        restored = _restored.get(_key(ids), 0)
        shared = max((n for old in _recent if (n := _lcp(ids, old)) < len(old) - CONTINUES), default=0)
        if not (_recent and len(_recent[-1]) == len(ids) and _lcp(ids, _recent[-1]) == len(ids)):
            _recent.append(ids)
    # On the stock interval grid (2048 = --prefill-step-size): a checkpoint off it splits a prefill chunk, which
    # changes kernel shapes and so the numerics of this request and of every one restored from it.
    bs = self.manager.block_size
    grid = max(bs, (self.manager.checkpoint_interval_tokens + bs - 1) // bs * bs)
    n = (shared - BACKOFF) // grid * grid
    n = _apc.adjust_prefix_to_text_suffix_boundary(token_ids, n, media_token_ids, max_prefix_tokens=lengths[-1])
    if n >= SHARED_MIN and n - restored >= SHARED_MIN and n not in lengths and n < lengths[-1]:
        log.info("APC shared-prefix checkpoint at %d/%d tokens (restored %d)", n, len(ids), restored)
        lengths = sorted([*lengths, n])
    return lengths


def _fingerprint(model_path):
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return None
    h = hashlib.sha256(str(path.resolve()).encode())
    for f in sorted(path.iterdir()):
        if f.suffix in (".safetensors", ".json"):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


_namespace = _apc.apc_disk_namespace


def apc_disk_namespace(model_path, **kwargs):
    if kwargs.get("weights_fingerprint") is None:
        kwargs["weights_fingerprint"] = _fingerprint(model_path)
    return _namespace(model_path, **kwargs)


def _prune_other_namespaces(own_dir, own_cap):
    files = []
    for d in own_dir.parent.iterdir():
        if d.is_dir() and d != own_dir:
            files += [(st.st_mtime, st.st_size, f) for f in d.glob("*.safetensors") if (st := f.stat())]
    budget = max(0, DISK_TOTAL - (own_cap or 0))
    total = sum(size for _, size, _ in files)
    removed = 0
    for _, size, f in sorted(files, key=lambda x: x[0]):
        if total <= budget:
            break
        f.unlink(missing_ok=True)
        total -= size
        removed += size
    if removed:
        log.info("APC disk: pruned %.1f GB of other models' caches; they now hold %.1f GB", removed / GB, total / GB)


_from_env = _apc.from_env


def from_env(*args, **kwargs):
    manager = _from_env(*args, **kwargs)
    if manager is not None and manager.disk is not None and DISK_TOTAL > 0:
        try:
            _prune_other_namespaces(manager.disk.dir, manager.disk.max_bytes)
        except OSError as e:
            log.warning("APC disk: pruning other namespaces failed: %s", e)
    return manager


APCCoordinator.lookup = lookup
APCCoordinator.checkpoint_lengths = checkpoint_lengths
_apc.apc_disk_namespace = apc_disk_namespace
_apc.from_env = from_env
