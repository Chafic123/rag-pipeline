from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness check: the process is up and serving requests."""
    return {"status": "ok"}
