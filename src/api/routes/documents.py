from fastapi import APIRouter, Depends

from ..models import DocumentUploadRequest, DocumentResponse
from ..responses import StandardResponse, success_response
from ...services.document_service import DocumentService
from ..dependencies import get_document_service

router = APIRouter()


@router.post("/documents",
             response_model=StandardResponse[DocumentResponse],
             )
def create_document(
        doc_in: DocumentUploadRequest,
        service: DocumentService = Depends(get_document_service),
):
    """创建文档，并异步进行分块、向量化和存储"""
    doc, chunks_count = service.create_document_with_chunks(
        name=doc_in.name,
        source_type=doc_in.source_type,
        source_content=doc_in.source_content,
    )
    return success_response(
        data=DocumentResponse(
            document_uid=doc.document_uid,
            name=doc.name,
            chunks_count=chunks_count,
        )
    )


@router.get(
    "/documents",
    response_model=StandardResponse[list[dict]],
)
def list_documents(service: DocumentService = Depends(get_document_service)):
    """获取文档列表"""
    docs = service.list_documents_for_api()
    return success_response(data=docs)
