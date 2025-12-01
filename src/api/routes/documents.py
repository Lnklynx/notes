from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select
from ..models import DocumentUploadRequest, DocumentResponse
from ...db.session import get_session
from ...services.document_service import (
    create_document_with_chunks,
    list_documents_for_api,
)
from ...services.runtime_service import build_document_components

router = APIRouter(prefix="/documents", tags=["documents"])


def get_services():
    """获取服务实例"""
    components = build_document_components()
    return components


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    req: DocumentUploadRequest,
    services: dict = Depends(get_services)
):
    """上传并处理文档"""
    try:
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

            return DocumentResponse(
                document_id=doc.document_uid,
                name=req.name,
                chunks_count=chunk_count,
                message="Document uploaded and processed successfully",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list")
async def list_documents():
    """列出所有文档（PostgreSQL）"""
    with get_session() as db:
        items = list_documents_for_api(db)
        return {"documents": items}

