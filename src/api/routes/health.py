from fastapi import APIRouter

from ..responses import StandardResponse, success_response

router = APIRouter(tags=["health"])


@router.get("/health", response_model=StandardResponse[dict])
async def health_check() -> StandardResponse:
    """健康检查"""
    response_data = {"status": "ok", "message": "Service is running"}
    return success_response(data=response_data)
