from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class BaseEmbedder(ABC):
    """向量化接口约定"""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """基于 SentenceTransformer 的向量化实现"""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


class QwenEmbeddingEmbedder(BaseEmbedder):
    """Qwen3-Embedding 系列向量化实现"""

    def __init__(
            self,
            model_id: str = "Qwen/Qwen3-Embedding-0.6B",
            model_dir: str | None = None,
    ):
        if model_dir is None:
            model_dir = snapshot_download(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: List[str]) -> list[list[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(**inputs)
            pooled = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy().tolist()

    def embed_text(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


class TextEmbedder(BaseEmbedder):
    """向量化层统一入口，按配置选择具体模型"""

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            model_dir: str | None = None,
    ):
        lower_name = model_name.lower()

        if model_dir:
            # 显式指定本地目录时，优先作为 Qwen 向量模型加载
            self._backend: BaseEmbedder = QwenEmbeddingEmbedder(
                model_id=model_name,
                model_dir=model_dir,
            )
        elif "qwen3-embedding" in lower_name or lower_name.startswith("qwen/"):
            self._backend = QwenEmbeddingEmbedder(model_id=model_name)
        else:
            self._backend = SentenceTransformerEmbedder(model_name=model_name)

    def embed_text(self, text: str) -> list[float]:
        return self._backend.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_batch(texts)
