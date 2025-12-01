from fastapi import APIRouter, HTTPException, Depends
from ..models import DocumentUploadRequest, DocumentResponse
from ...embedding.chunker import DocumentChunker
from ...embedding.embedder import TextEmbedder
from ...embedding.vector_store import VectorStore
import hashlib
import uuid

router = APIRouter(prefix="/documents", tags=["documents"])

# 依赖注入占位
def get_services():
    """获取服务实例（实际应该从容器注入）"""
    return {
        "chunker": DocumentChunker(),
        "embedder": TextEmbedder(),
        "vector_store": VectorStore()
    }


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    req: DocumentUploadRequest,
    services: dict = Depends(get_services)
):
    """上传并处理文档"""
    try:
        doc_id = str(uuid.uuid4())
        
        # 获取文本内容
        if req.source_type == "text":
            text = req.content
        elif req.source_type == "url":
            # TODO: 实现 URL 加载逻辑
            text = req.content
        else:
            raise ValueError(f"Unsupported source type: {req.source_type}")
        
        # 分块
        chunker = services["chunker"]
        chunks = chunker.chunk(text)
        
        # 向量化
        embedder = services["embedder"]
        embeddings = embedder.embed_batch(chunks)
        
        # 存储
        vector_store = services["vector_store"]
        vector_store.create_collection("documents")
        
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source_id": doc_id, "name": req.name} for _ in chunks]
        )
        vector_store.persist()
        
        return DocumentResponse(
            document_id=doc_id,
            name=req.name,
            chunks_count=len(chunks),
            message="Document uploaded and processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list")
async def list_documents():
    """列出所有文档（占位实现）"""
    return {"documents": []}

