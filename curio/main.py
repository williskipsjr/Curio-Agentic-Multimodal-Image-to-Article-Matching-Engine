"""FastAPI application entry point."""

from fastapi import FastAPI

from curio import __version__

app = FastAPI(
    title="Curio",
    description="Agentic Multimodal Image-to-Article Matching Engine",
    version=__version__,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}
