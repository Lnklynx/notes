import chromadb
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """向量数据库封装（ChromaDB）

    当前使用 Docker 部署的 Chroma HTTP 服务，而不是本地文件持久化。
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        # 通过 HTTP 客户端连接 Chroma 服务
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = None

    def create_collection(self, collection_name: str):
        """创建或获取集合"""
        logger.info(f"[VectorStore] 创建/获取集合: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        count = self.collection.count() if self.collection else 0
        logger.info(f"[VectorStore] 集合 '{collection_name}' 当前包含 {count} 个向量")

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ):
        """添加文档到向量库"""
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or []
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        """检索相似文档，可选 where 过滤"""
        if not self.collection:
            raise ValueError("Collection not initialized.")
        
        logger.debug(f"[VectorStore] 执行向量检索 | top_k: {top_k} | where: {where}")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where or {},
            )
            
            doc_count = len(results.get("documents", [[]])[0]) if results.get("documents") else 0
            logger.info(f"[VectorStore] 检索完成 | 返回 {doc_count} 个结果")
            
            return results
        except Exception as e:
            logger.error(f"[VectorStore] 检索失败: {e}")
            raise

    def delete_by_source(self, source_id: str):
        """删除某个来源的所有文档"""
        if not self.collection:
            raise ValueError("Collection not initialized.")
        
        # 按元数据过滤删除
        self.collection.delete(
            where={"source_id": source_id}
        )

    def persist(self):
        """持久化数据（PersistentClient 自动持久化，此方法保留以保持兼容性）"""
        pass

