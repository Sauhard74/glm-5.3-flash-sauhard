import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from .config import ServeConfig
from .engine import Engine

logger = logging.getLogger("serve.app")

REMOVED_ROUTES = {
    ("/v1/settings", "PATCH"),
    ("/settings", "PATCH"),
    ("/v1/models", "GET"),
    ("/models", "GET"),
    ("/unload", "POST"),
}
NON_INFERENCE_POSTS = {"/unload", "/v1/cache/reset", "/cache/reset", "/serve/load", "/serve/unload"}


class ActivityMiddleware:
    """Counts in-flight inference requests (through the end of a stream) and
    loads the pinned model off the event loop before the protocol layer runs."""

    def __init__(self, app, engine: Engine):
        self.app = app
        self.engine = engine

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.engine.cfg.api_key:
            scope = _accept_x_api_key(scope)
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] in NON_INFERENCE_POSTS
            or not self._authorized(scope)
        ):
            return await self.app(scope, receive, send)
        self.engine.begin()
        try:
            if not self.engine.is_loaded():
                try:
                    await asyncio.to_thread(self.engine.ensure_loaded)
                except Exception as e:
                    logger.exception("Model load failed")
                    response = JSONResponse(
                        status_code=getattr(e, "status_code", 503),
                        content=_error_body(scope["path"], str(e)),
                    )
                    return await response(scope, receive, send)
            await self.app(scope, receive, send)
        finally:
            self.engine.end()


    def _authorized(self, scope) -> bool:
        key = self.engine.cfg.api_key
        if not key:
            return True
        headers = dict(scope.get("headers") or [])
        supplied = headers.get(b"authorization", b"").decode("latin-1")
        return secrets.compare_digest(supplied, f"Bearer {key}")


def _accept_x_api_key(scope):
    """Anthropic clients send the key as x-api-key; mlx-vlm only checks Bearer auth."""
    headers = list(scope.get("headers") or [])
    names = {k for k, _ in headers}
    if b"authorization" in names:
        return scope
    key = next((v for k, v in headers if k == b"x-api-key"), None)
    if key is None:
        return scope
    return {**scope, "headers": headers + [(b"authorization", b"Bearer " + key)]}


def _error_body(path: str, message: str) -> dict:
    if path.rstrip("/").endswith("messages") or "count_tokens" in path:
        return {"type": "error", "error": {"type": "api_error", "message": message}}
    return {"error": {"message": message, "type": "server_error", "code": None}}


def create_app(cfg: ServeConfig, engine: Engine = None) -> FastAPI:
    engine = engine or Engine(cfg)
    engine.install()
    require_key = engine.app_mod._require_management_api_key

    @asynccontextmanager
    async def lifespan(_app):
        logger.info("serve config: %s", json.dumps(cfg.to_dict(), default=str))
        if cfg.preload:
            await asyncio.to_thread(engine.ensure_loaded)
        engine.start()
        try:
            yield
        finally:
            await asyncio.to_thread(engine.shutdown)

    app = FastAPI(title="local-ai serve", lifespan=lifespan)

    def models(request: Request):
        require_key(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": cfg.name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "loaded": engine.is_loaded(),
                    "max_context": cfg.max_context,
                }
            ],
        }

    def status(request: Request):
        require_key(request)
        return engine.status()

    async def load(request: Request):
        require_key(request)
        await asyncio.to_thread(engine.ensure_loaded)
        return engine.status()

    async def unload(request: Request):
        require_key(request)
        force = request.query_params.get("force") in ("1", "true")
        done = await asyncio.to_thread(engine.unload, "requested", force=force)
        if not done and engine.is_loaded():
            return JSONResponse(status_code=409, content={"status": "busy", "inflight": engine.inflight})
        return {"status": "unloaded" if done else "not_loaded"}

    # Registered first so they shadow mlx-vlm's versions of the same paths.
    for path in ("/v1/models", "/models"):
        app.add_api_route(path, models, methods=["GET"])
    app.add_api_route("/serve/status", status, methods=["GET"])
    app.add_api_route("/serve/load", load, methods=["POST"])
    for path in ("/serve/unload", "/unload"):
        app.add_api_route(path, unload, methods=["POST"])

    app.include_router(engine.app_mod.inference_router)
    app.router.routes.extend(
        r
        for r in engine.srv.app.router.routes
        if isinstance(r, APIRoute)
        and not any((r.path, m) in REMOVED_ROUTES for m in r.methods or ())
    )

    app.add_middleware(ActivityMiddleware, engine=engine)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.engine = engine
    return app
