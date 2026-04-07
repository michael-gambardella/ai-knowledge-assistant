from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Confirms the API is running.
    Production systems use this for load balancer checks and uptime monitoring.
    """
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        version="1.0.0",
    )
