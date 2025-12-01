from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """向量化处理"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """单条向量化"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
