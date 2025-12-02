from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar('T')


class StandardResponse(BaseModel, Generic[T]):
    """标准 REST 响应结构"""
    code: int
    message: str
    data: Optional[T] = None

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {}
            }
        }


def success_response(data=None, message: str = "success") -> StandardResponse:
    """成功响应"""
    return StandardResponse(code=200, message=message, data=data)


def error_response(code: int, message: str, data=None) -> StandardResponse:
    """错误响应"""
    return StandardResponse(code=code, message=message, data=data)

