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

    def delete_document_by_uid(self, document_uid: str) -> bool:
        """根据 document_uid 删除文档、分块及向量"""

        # 1. 查找文档
        doc = self.db.exec(
            select(Document).where(
                Document.document_uid == document_uid,
                Document.is_deleted == False  # noqa: E712
            )
        ).first()

        if not doc:
            return False

        # 2. 删除向量库中该文档的所有 chunks（物理删除，向量库不支持逻辑删除）
        self.vector_store.delete_by_document_uid(document_uid)

        # 3. 逻辑删除数据库中该文档的所有 chunks
        chunks_to_delete = self.db.exec(
            select(DocumentChunk).where(
                DocumentChunk.document_id == doc.id,
                DocumentChunk.is_deleted == False  # noqa: E712
            )
        ).all()
        for chunk in chunks_to_delete:
            chunk.is_deleted = True
            self.db.add(chunk)

        # 4. 逻辑删除文档记录
        doc.is_deleted = True
        self.db.add(doc)
        self.db.commit()

        return True

    def batch_delete_documents(self, document_uids: list[str]) -> dict:
        """批量删除文档、分块及向量
        
        Returns:
            dict: 包含 success_count, failed_count, failed_uids 的字典
        """
        success_count = 0
        failed_uids = []

        for document_uid in document_uids:
            try:
                # 每个文档的删除操作是独立的，如果失败不影响其他文档
                deleted = self.delete_document_by_uid(document_uid)
                if deleted:
                    success_count += 1
                else:
                    failed_uids.append(document_uid)
            except Exception as e:
                # 如果删除过程中出现异常，回滚当前事务并记录失败的文档
                failed_uids.append(document_uid)
                try:
                    self.db.rollback()
                except Exception:
                    # 如果回滚也失败，忽略（可能事务已经提交或不存在）
                    pass
        return {
            "success_count": success_count,
            "failed_count": len(failed_uids),
            "failed_uids": failed_uids,
        }

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
