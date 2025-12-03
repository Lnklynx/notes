from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_document_service
from ..models import (
    DocumentUploadRequest, DocumentResponse, BatchDeleteRequest, BatchDeleteResponse,
)
from ..responses import success_response
from ...services.document_service import DocumentService

router = APIRouter()


@router.post("/documents",
             summary="创建文档",
             description="创建文档，自动分块、向量化并存入向量库")
def create_document(
        doc_in: DocumentUploadRequest,
        service: DocumentService = Depends(get_document_service),
):
    doc, chunks_count = service.create_document_with_chunks(
        name=doc_in.name,
        source_type=doc_in.source_type,
        source_content=doc_in.content,
    )
    response = DocumentResponse(
        document_uid=doc.document_uid,
        name=doc.name,
        chunks_count=chunks_count,
    )
    return success_response(data=response)


@router.get("/documents/list",
            summary="获取文档列表",
            description="获取所有已创建的文档列表")
def list_documents(service: DocumentService = Depends(get_document_service)):
    docs = service.list_documents_for_api()
    return success_response(data={"items": docs})


@router.delete("/documents/{document_uid}",
               summary="删除文档",
               description="根据 document_uid 删除文档，同时删除相关的 chunks 和向量库中的数据")
def delete_document(
        document_uid: str,
        service: DocumentService = Depends(get_document_service),
):
    """删除指定文档"""
    deleted = service.delete_document_by_uid(document_uid)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"文档不存在: {document_uid}")
    return success_response(message=f"文档 {document_uid} 已成功删除")


@router.post("/documents/batch-delete",
             summary="批量删除文档",
             description="批量删除多个文档，同时删除相关的 chunks 和向量库中的数据")
def batch_delete_documents(
        request: BatchDeleteRequest,
        service: DocumentService = Depends(get_document_service),
):
    """批量删除文档"""
    result = service.batch_delete_documents(request.document_uids)
    response = BatchDeleteResponse(
        success_count=result["success_count"],
        failed_count=result["failed_count"],
        failed_uids=result["failed_uids"],
    )
    return success_response(
        data=response,
        message=f"批量删除完成：成功 {result['success_count']} 个，失败 {result['failed_count']} 个"
    )
