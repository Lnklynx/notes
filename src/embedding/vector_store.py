import chromadb
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """向量数据库封装（ChromaDB）

    当前使用 Docker 部署的 Chroma HTTP 服务，而不是本地文件持久化。
    """

    DEFAULT_COLLECTION_NAME = "documents"

    def __init__(self, host: str = "localhost", port: int = 8000):
        # 通过 HTTP 客户端连接 Chroma 服务
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = None
        self._collection_name = None

    def _ensure_collection(self, collection_name: str | None = None):
        """确保 collection 已初始化（懒加载）"""
        collection_name = collection_name or self._collection_name or self.DEFAULT_COLLECTION_NAME
        if self.collection is None or self._collection_name != collection_name:
            self.create_collection(collection_name)

    def create_collection(self, collection_name: str):
        """创建或获取集合"""
        logger.info(f"[VectorStore] 创建/获取集合: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._collection_name = collection_name
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
        self._ensure_collection()
        
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
        self._ensure_collection()
        
        logger.debug(f"[VectorStore] 执行向量检索 | top_k: {top_k} | where: {where}")
        
        try:
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            # ChromaDB 不接受空字典，只在有有效条件时才添加 where
            if where:
                query_kwargs["where"] = where
            
            results = self.collection.query(**query_kwargs)
            
            doc_count = len(results.get("documents", [[]])[0]) if results.get("documents") else 0
            logger.info(f"[VectorStore] 检索完成 | 返回 {doc_count} 个结果")
            
            return results
        except Exception as e:
            logger.error(f"[VectorStore] 检索失败: {e}")
            raise

    def search_by_metadata(
        self,
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        """按元数据过滤查询文档（不需要向量检索）"""
        self._ensure_collection()
        
        logger.debug(f"[VectorStore] 执行元数据查询 | top_k: {top_k} | where: {where}")
        
        try:
            if not where:
                logger.warning("[VectorStore] 元数据查询未提供 where 条件，返回空结果")
                return {
                    "documents": [[]],
                    "distances": [[]],
                    "metadatas": [[]],
                    "ids": [[]],
                }
            
            results = self.collection.get(
                where=where,
                limit=top_k,
            )
            
            doc_count = len(results.get("documents", [])) if results.get("documents") else 0
            logger.info(f"[VectorStore] 元数据查询完成 | 返回 {doc_count} 个结果")
            
            return {
                "documents": [results.get("documents", [])],
                "distances": [[]],  # 元数据查询没有距离
                "metadatas": [results.get("metadatas", [])],
                "ids": [results.get("ids", [])],
            }
        except Exception as e:
            logger.error(f"[VectorStore] 元数据查询失败: {e}")
            raise

    def delete_by_source(self, source_id: str):
        """删除某个来源的所有文档"""
        self._ensure_collection()
        
        # 按元数据过滤删除
        self.collection.delete(
            where={"source_id": source_id}
        )

    def persist(self):
        """持久化数据（PersistentClient 自动持久化，此方法保留以保持兼容性）"""
        pass

