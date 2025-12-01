from __future__ import annotations

from typing import TypedDict

from ..config import get_settings, Settings
from ..llm.factory import create_llm
from ..llm.base import BaseLLM
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..embedding.chunker import DocumentChunker


class LLMComponents(TypedDict):
    llm: BaseLLM
    embedder: TextEmbedder
    vector_store: VectorStore


class DocumentComponents(TypedDict):
    chunker: DocumentChunker
    embedder: TextEmbedder
    vector_store: VectorStore


def build_llm_components(settings: Settings | None = None) -> LLMComponents:
    """构建 LLM 相关组件（LLM + Embedding + VectorStore）"""
    settings = settings or get_settings()

    provider = settings.llm_provider.lower()
    api_key = ""
    if provider == "openai":
        api_key = settings.openai_api_key
    elif provider == "qwen":
        api_key = settings.qwen_api_key

    llm = create_llm(
        provider,
        api_key=api_key,
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
    )
    embedder = TextEmbedder(settings.embedding_model)
    vector_store = VectorStore(settings.vector_db_path)

    return {
        "llm": llm,
        "embedder": embedder,
        "vector_store": vector_store,
    }


def build_document_components(settings: Settings | None = None) -> DocumentComponents:
    """构建文档处理相关组件（Chunker + Embedding + VectorStore）"""
    settings = settings or get_settings()

    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = TextEmbedder(settings.embedding_model)
    vector_store = VectorStore(settings.vector_db_path)

    return {
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
    }
