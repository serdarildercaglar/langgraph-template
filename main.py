"""
Main FastAPI application entry point.
Provides the production-ready API server.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import v1_router
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()

    # Initialize Langfuse
    from observability import get_langfuse_manager
    langfuse = get_langfuse_manager()
    if langfuse.enabled:
        print(f"✓ Langfuse enabled (host: {settings.langfuse.host})")
    else:
        print("⚠ Langfuse disabled")

    # Import tools to trigger registration
    import tools  # noqa: F401

    from agents import get_all_agents
    from tools import get_all_tools

    print(f"✓ Registered {len(get_all_agents())} agents")
    print(f"✓ Registered {len(get_all_tools())} tools")

    print(f"✓ Server starting on {settings.api.host}:{settings.api.port}")

    yield

    # Shutdown
    print("Shutting down...")

    # Flush Langfuse
    if langfuse.enabled:
        langfuse.flush()

    # Close Milvus connection
    from retrieval import get_milvus_client
    await get_milvus_client().close()

    # Close embedding service
    from retrieval import get_embedding_service
    await get_embedding_service().close()

    print("✓ Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        debug=settings.api.debug,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(v1_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
