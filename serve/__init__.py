"""Memory-capped OpenAI/Anthropic-compatible server for one local MLX model."""

from .config import ServeConfig

__all__ = ["ServeConfig", "create_app"]


def create_app(cfg: ServeConfig):
    from .app import create_app as _create_app

    return _create_app(cfg)
