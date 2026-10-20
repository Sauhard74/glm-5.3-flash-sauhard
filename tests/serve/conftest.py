import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from serve.config import ServeConfig
from serve.testing import build_tiny

from serve_helpers import ARCHS


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    models = {}
    for arch in ARCHS:
        try:
            models[arch] = str(build_tiny(arch, tmp_path_factory.mktemp(f"tiny-{arch}")))
        except Exception as e:  # no network and no cached tokenizer files
            pytest.skip(f"cannot build tiny {arch} checkpoint: {e}")
    return models


@pytest.fixture
def serve_client(tmp_path, monkeypatch):
    """serve_client(model, **ServeConfig overrides) -> TestClient with lifespan run."""
    stack = contextlib.ExitStack()
    hook = sys.unraisablehook

    def unraisable(u):
        # mlx-vlm's BatchGenerator.__del__ syncs its GPU stream even when the GC
        # runs it on another thread; pytest's hook would pin the generator (and
        # the model weights) in memory.
        if "BatchGenerator.__del__" in repr(getattr(u, "object", "")) or "There is no Stream" in str(u.exc_value):
            return
        hook(u)

    monkeypatch.setattr(sys, "unraisablehook", unraisable)

    def make(model, **overrides):
        from serve.app import create_app

        overrides.setdefault("prefix_cache_dir", str(tmp_path / "prefix-cache"))
        overrides.setdefault("idle_unload_s", 0)
        cfg = ServeConfig(model=model, **overrides)
        return stack.enter_context(TestClient(create_app(cfg)))

    yield make
    stack.close()


class _Forcer:
    def __init__(self, ids):
        self.ids = ids
        self.i = 0

    def __call__(self, tokens, logits):
        if self.i >= len(self.ids):
            return logits
        target = self.ids[self.i]
        self.i += 1
        hit = mx.arange(logits.shape[-1]) == target
        return mx.where(hit, mx.zeros_like(logits), mx.full(logits.shape, -1e9, logits.dtype))


@pytest.fixture
def force_output(monkeypatch):
    """force_output(client, text): the next generations emit exactly `text` then EOS."""
    from mlx_vlm.server.generation import ResponseGenerator

    state = {"ids": None}
    orig = ResponseGenerator._make_logits_processors

    def patched(self, args, input_ids=None):
        procs = list(orig(self, args, input_ids))
        if state["ids"] is not None:
            procs.append(_Forcer(state["ids"]))
        return procs

    monkeypatch.setattr(ResponseGenerator, "_make_logits_processors", patched)

    def set_text(client, text):
        rg = client.app.state.engine.srv.runtime.response_generator
        eos = sorted(rg.stop_tokens)[0]
        state["ids"] = rg.tokenizer.encode(text, add_special_tokens=False) + [eos]

    return set_text
