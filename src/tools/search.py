from .base import Tool
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore


class VectorSearchTool(Tool):
    """向量检索工具"""

    name = "vector_search"
    description = "Search for relevant documents based on query"

    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def execute(self, query: str, top_k: int = 5) -> dict:
        """执行检索"""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return {
            "documents": results.get("documents", [[]]),
            "distances": results.get("distances", [[]]),
            "metadatas": results.get("metadatas", [[]])
        }

