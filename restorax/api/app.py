import logging
import logging.config
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from restorax.api.middleware import RequestIDMiddleware, TimingMiddleware
from restorax.api.routers import jobs, models, pipelines, ws
from restorax.api.routers.health import router as health_router
from restorax.config import settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from restorax.db.session import create_tables
    await create_tables()
    yield


def create_app() -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    app = FastAPI(
        title="RestoraX",
        description="Modern AI video restoration platform",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(pipelines.router)
    app.include_router(ws.router)

    # Prometheus metrics — optional dependency
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    except ImportError:
        pass  # prometheus-fastapi-instrumentator not installed — skip silently

    return app


app = create_app()
