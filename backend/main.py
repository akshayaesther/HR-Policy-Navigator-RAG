"""FastAPI application entrypoint for Enterprise HR & Policy Navigator."""

from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise HR & Policy Navigator with Evaluated Hybrid Search",
    version="1.0.0",
    debug=settings.debug,
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Endpoint")
def read_root() -> Dict[str, Any]:
    """Provide root status and service metadata."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", summary="Health Check")
def health_check() -> Dict[str, Any]:
    """Check API server health status."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
