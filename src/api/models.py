from enum import Enum
from typing import Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    content: str = Field(..., description="文档内容或URL")
    source_type: str = Field(..., description="文件类型：url/text/file")
    name: str = Field(..., description="文档名称")


class DocumentResponse(BaseModel):
    """文档创建结果"""
    document_uid: str
    name: str
    chunks_count: int


class DocumentInfo(BaseModel):
    document_uid: str
    name: str
    source_type: str
    status: str
    version: int
    created_at: datetime


class ResourceMode(str, Enum):
    """资源模式枚举"""
    DOCUMENTS = "documents"  # 明确指定一个或多个文档
    KNOWLEDGE_BASE = "knowledge_base"  # 指定知识库
    PROJECT = "project"  # 指定项目
    ALL = "all"  # 在所有可触及的资源中检索
    AUTO = "auto"  # 由 Agent 自行判断


class ResourceScope(BaseModel):
    """
    定义用户请求的资源范围。
    用于指导 Agent 在哪个边界内进行思考和信息检索。
    """
    mode: ResourceMode = Field(
        default=ResourceMode.AUTO,
        description="资源模式，决定 Agent 的检索策略"
    )
    document_ids: list[str] = Field(
        default_factory=list,
        description="文档 ID 列表，在 'documents' 模式下使用"
    )
    kb_ids: list[str] = Field(
        default_factory=list,
        description="知识库 ID 列表，在 'knowledge_base' 模式下使用"
    )
    project_ids: list[str] = Field(
        default_factory=list,
        description="项目 ID 列表，在 'project' 模式下使用"
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="附加的元数据过滤条件，如标签、日期等"
    )


class ChatRequest(BaseModel):
    """
    对话请求模型，代表用户的单次输入意图。
    """
    conversation_uid: str = Field(..., description="会话的唯一标识")
    message: str = Field(..., description="用户本次发送的消息")
    scope: ResourceScope = Field(
        default_factory=ResourceScope,  # 默认自动模式
        description="本次请求的资源范围"
    )
    stream: bool = Field(default=False, description="是否启用流式响应")


class ChatResponse(BaseModel):
    """对话响应"""
    conversation_uid: str
    user_message: str
    answer: str
    # 返回具体的参考内容片段
    retrieved_content: list[str] = Field(default_factory=list, description="参考内容片段")
    # 返回本次交互引用的源文档 ID 列表
    source_documents: list[str] = Field(default_factory=list, description="引用的源文档UID列表")


class HistoryResponse(BaseModel):
    """对话历史"""
    conversation_uid: str
    messages: list[dict]


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None


class DocumentListResponse(BaseModel):
    items: list[DocumentInfo]


class VectorQueryRequest(BaseModel):
    where: dict[str, Any] = Field({}, description="Metadata filter for the query. Empty dict for no filter.")
    limit: int = Field(10, description="Maximum number of documents to return", ge=1, le=100)


class VectorQueryItem(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any]


class VectorQueryData(BaseModel):
    items: list[VectorQueryItem]


class BatchDeleteRequest(BaseModel):
    """批量删除文档请求"""
    document_uids: list[str] = Field(..., description="要删除的文档 UID 列表", min_length=1)


class BatchDeleteResponse(BaseModel):
    """批量删除文档响应"""
    success_count: int = Field(..., description="成功删除的数量")
    failed_count: int = Field(..., description="删除失败的数量")
    failed_uids: list[str] = Field(default_factory=list, description="删除失败的文档 UID 列表")


# ========== 向量库 CRUD 相关模型 ==========

class VectorAddRequest(BaseModel):
    """添加向量数据请求"""
    ids: list[str] = Field(..., description="向量 ID 列表", min_length=1)
    embeddings: list[list[float]] = Field(..., description="向量嵌入列表", min_length=1)
    documents: list[str] = Field(..., description="文档内容列表", min_length=1)
    metadatas: list[dict[str, Any]] | None = Field(default=None, description="元数据列表")
    collection_name: str | None = Field(default=None, description="集合名称，默认使用 'documents'")


class VectorUpdateRequest(BaseModel):
    """更新向量数据请求"""
    ids: list[str] = Field(..., description="要更新的向量 ID 列表", min_length=1)
    embeddings: list[list[float]] | None = Field(default=None, description="新的向量嵌入列表")
    documents: list[str] | None = Field(default=None, description="新的文档内容列表")
    metadatas: list[dict[str, Any]] | None = Field(default=None, description="新的元数据列表")
    collection_name: str | None = Field(default=None, description="集合名称，默认使用 'documents'")


class VectorDeleteRequest(BaseModel):
    """删除向量数据请求"""
    ids: list[str] | None = Field(default=None, description="要删除的向量 ID 列表")
    where: dict[str, Any] | None = Field(default=None, description="按元数据过滤条件删除")
    collection_name: str | None = Field(default=None, description="集合名称，默认使用 'documents'")


class VectorSearchRequest(BaseModel):
    """向量检索请求"""
    query_embedding: list[float] = Field(..., description="查询向量", min_length=1)
    top_k: int = Field(5, description="返回最相似的前 K 个结果", ge=1, le=100)
    where: dict[str, Any] | None = Field(default=None, description="元数据过滤条件")
    collection_name: str | None = Field(default=None, description="集合名称，默认使用 'documents'")


class VectorCollectionInfo(BaseModel):
    """向量集合信息"""
    name: str = Field(..., description="集合名称")
    count: int = Field(..., description="向量数量")


class VectorCollectionListResponse(BaseModel):
    """向量集合列表响应"""
    collections: list[VectorCollectionInfo] = Field(..., description="集合列表")
