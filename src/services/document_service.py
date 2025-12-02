from __future__ import annotations

from typing import Sequence

from sqlmodel import Session, select

from ..db.models import Document, DocumentChunk
from ..embedding.chunker import DocumentChunker
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..tools.document_loader import DocumentLoader


class DocumentService:
    def __init__(
            self,
            db: Session,
            chunker: DocumentChunker,
            embedder: TextEmbedder,
            vector_store: VectorStore,
    ):
        self.db = db
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def create_document_with_chunks(
            self,
            *,
            name: str,
            source_type: str,
            source_content: str,
    ) -> tuple[Document, int]:
        """创建文档及分块，并写入向量库"""

        # 1. 根据来源类型加载原始文本（URL / 文件 / 直接文本）
        text = DocumentLoader.load(source_content, source_type=source_type)

        # 2. 分块与向量化
        chunks = self.chunker.chunk(text)
        embeddings = self.embedder.embed_batch(chunks)

        # 3. 文档记录（保留原始来源 content，便于追溯）
        doc = Document(
            name=name,
            source_type=source_type,
            source=source_content,
            user_id=None,
        )
        self.db.add(doc)
        self.db.flush()

        # 4. 分块记录
        chunk_rows: list[DocumentChunk] = []
        for idx, content in enumerate(chunks):
            row = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                content=content,
            )
            self.db.add(row)
            chunk_rows.append(row)

        # 5. 向量库写入
        self.vector_store.create_collection("documents")

        chunk_ids = [row.chunk_uid for row in chunk_rows]
        metadatas = [
            {
                "document_uid": doc.document_uid,
                "document_id": doc.id,
                "chunk_uid": row.chunk_uid,
                "chunk_index": row.chunk_index,
                "name": name,
            }
            for row in chunk_rows
        ]

        self.vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return doc, len(chunks)

    def list_documents_for_api(self) -> list[dict]:
        """列出文档信息，供 API 使用"""

        docs: Sequence[Document] = self.db.exec(
            select(Document).where(Document.is_deleted == False)  # noqa: E712
        ).all()

        return [
            {
                "document_uid": d.document_uid,
                "name": d.name,
                "source_type": d.source_type,
                "status": d.status,
                "version": d.version,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ]
