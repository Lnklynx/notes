from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "Service is running"}

