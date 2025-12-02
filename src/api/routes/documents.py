from typing import List

from fastapi import APIRouter, Depends

from ..dependencies import DocumentComponents, build_document_components
from ..models import (
    DocumentResponse,
    DocumentUploadRequest,
)
from ..responses import StandardResponse, success_response
from ...db.session import get_session
from ...services.document_service import (
    create_document_with_chunks,
    list_documents_for_api,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_services() -> DocumentComponents:
    """依赖注入：获取文档处理服务"""
    components = build_document_components()
    return components


@router.post(
    "/upload",
    response_model=StandardResponse[DocumentResponse],
    summary="上传并处理文档",
)
async def upload_document(
        req: DocumentUploadRequest,
        services: DocumentComponents = Depends(get_document_services),
) -> StandardResponse:
    """上传文档，对其进行分块、向量化并存储"""
    with get_session() as db:
        doc, chunk_count = create_document_with_chunks(
            db=db,
            name=req.name,
            source_type=req.source_type,
            source_content=req.content,
            chunker=services["chunker"],
            embedder=services["embedder"],
            vector_store=services["vector_store"],
        )

    response_data = DocumentResponse(
        document_id=doc.document_uid,
        name=req.name,
        chunks_count=chunk_count,
        message="Document uploaded and processed successfully",
    )
    return success_response(data=response_data)


@router.get(
    "/list",
    response_model=StandardResponse[List[dict]],
    summary="列出所有文档",
)
async def list_documents() -> StandardResponse:
    """获取系统中所有已处理的文档列表"""
    with get_session() as db:
        items = list_documents_for_api(db)
        return success_response(data=items)
