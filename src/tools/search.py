from .base import Tool
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorSearchTool(Tool):
    """向量检索工具"""

    name = "vector_search"
    description = "Search for relevant documents based on query. Use this tool when you need to retrieve information from documents to answer the user's question."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top documents to retrieve (default: 5)",
                "default": 5
            },
            "document_uid": {
                "type": "string",
                "description": "Optional document identifier to filter results to a specific document"
            }
        },
        "required": ["query"]
    }

    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def execute(
        self,
        query: str,
        top_k: int = 5,
        document_uid: str | None = None,
    ) -> dict:
        """执行检索，可选按文档过滤"""
        logger.info(f"[VectorSearchTool] 开始检索 | query: {query[:100]}... | top_k: {top_k} | document_uid: {document_uid}")
        
        query_embedding = self.embedder.embed_text(query)
        logger.debug(f"[VectorSearchTool] 向量化完成 | 维度: {len(query_embedding)}")

        where = {"document_uid": document_uid} if document_uid else None
        if where:
            logger.info(f"[VectorSearchTool] 使用文档过滤: {where}")
        
        results = self.vector_store.search(
            query_embedding, top_k=top_k, where=where
        )

        doc_list = results.get("documents", [[]])
        doc_count = len(doc_list[0]) if doc_list and doc_list[0] else 0
        distances = results.get("distances", [[]])
        
        logger.info(f"[VectorSearchTool] 检索结果 | 返回 {doc_count} 个文档片段")
        if doc_count > 0 and distances and distances[0]:
            logger.debug(f"[VectorSearchTool] 相似度范围: {min(distances[0]):.4f} ~ {max(distances[0]):.4f}")
        elif doc_count == 0:
            logger.warning(f"[VectorSearchTool] ⚠️ 检索结果为空！可能原因：1) Chroma 集合为空 2) where 过滤后无匹配 3) 查询向量与现有向量差异过大")

        return {
            "documents": doc_list,
            "distances": distances,
            "metadatas": results.get("metadatas", [[]]),
        }
