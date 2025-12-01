import chromadb


class VectorStore:
    """向量数据库封装（ChromaDB）"""

    def __init__(self, db_path: str = "./data/vectordb"):
        # 开发阶段使用本地文件持久化。
        # 如果后续切换到 Docker Chroma HTTP 服务，可在此替换为 HttpClient。
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None

    def create_collection(self, collection_name: str):
        """创建或获取集合"""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

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
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or {},
        )
        return results

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

