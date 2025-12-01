from pydantic import BaseModel, Field
from typing import Optional


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    content: str = Field(..., description="文档内容或URL")
    source_type: str = Field(..., description="文件类型：url/text/file")
    name: str = Field(..., description="文档名称")


class DocumentResponse(BaseModel):
    """文档响应"""
    document_id: str
    name: str
    chunks_count: int
    message: str


class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(..., description="会话ID")
    document_id: str = Field(..., description="文档ID")
    message: str = Field(..., description="用户问题")
    stream: bool = Field(default=False, description="是否流式返回")


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str
    document_id: str
    user_message: str
    answer: str
    documents: list[str] = Field(default_factory=list, description="参考文档片段")


class HistoryResponse(BaseModel):
    """对话历史"""
    session_id: str
    messages: list[dict]


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None

