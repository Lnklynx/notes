from __future__ import annotations

from typing import TypedDict

from ..config import Settings, get_settings
from ..embedding.chunker import DocumentChunker
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..llm.base import BaseLLM
from ..llm.factory import create_llm


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
    model = settings.llm_model

    if provider == "openai":
        api_key = settings.openai_api_key
        model = settings.llm_model
    elif provider == "ollama":
        model = settings.ollama_model
    elif provider == "qwen":
        api_key = settings.qwen_api_key
        model = settings.qwen_model

    llm = create_llm(
        provider,
        api_key=api_key,
        model=model,
        base_url=settings.ollama_base_url,
    )
    embedder = TextEmbedder(
        model_name=settings.embedding_model,
        model_dir=settings.embedding_model_dir,
    )
    # 使用 Docker Chroma HTTP 服务
    vector_store = VectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

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
    embedder = TextEmbedder(
        model_name=settings.embedding_model,
        model_dir=settings.embedding_model_dir,
    )
    vector_store = VectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    return {
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
    }
