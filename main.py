"""
Main FastAPI application entry point.
Provides the production-ready API server.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents import get_all_agents
from api import v1_router
from config.settings import get_settings
from observability import get_langfuse_manager
from retrieval import get_embedding_service, get_milvus_client
from tools import get_all_tools


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings = get_settings()

    # Initialize services
    langfuse = get_langfuse_manager()
    print(f"{'✓' if langfuse.enabled else '⚠'} Langfuse {'enabled' if langfuse.enabled else 'disabled'}")

    # Trigger tool registration by accessing the module
    import tools  # noqa: F401

    print(f"✓ Registered {len(get_all_agents())} agents")
    print(f"✓ Registered {len(get_all_tools())} tools")
    print(f"✓ Server starting on {settings.api.host}:{settings.api.port}")

    yield

    # Shutdown
    print("Shutting down...")
    if langfuse.enabled:
        langfuse.flush()
    await get_milvus_client().close()
    await get_embedding_service().close()
    print("✓ Shutdown complete")


settings = get_settings()

app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    debug=settings.api.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
