from functools import lru_cache
from fastapi import Depends
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_session
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..embedding.chunker import DocumentChunker
from ..llm.base import BaseLLM
from ..llm.factory import create_llm
from ..agent.factory import AgentFactory
from ..services.runtime_service import AgentRuntimeService
from ..services.conversation_service import ConversationService
from ..services.chat_service import ChatApplicationService
from ..services.document_service import DocumentService
from ..tools.base import ToolRegistry
from ..tools.search import VectorSearchTool


@lru_cache()
def get_llm() -> BaseLLM:
    """依赖注入：获取 LLM 实例（进程级单例）"""
    settings: Settings = get_settings()
    provider = settings.llm_provider.lower()
    api_key_map = {
        "openai": settings.openai_api_key,
        "qwen": settings.qwen_api_key,
    }
    model_map = {
        "openai": settings.llm_model,
        "ollama": settings.ollama_model,
        "qwen": settings.qwen_model,
    }
    return create_llm(
        provider,
        api_key=api_key_map.get(provider, ""),
        model=model_map.get(provider, settings.llm_model),
        base_url=settings.ollama_base_url,
    )


@lru_cache()
def get_embedder() -> TextEmbedder:
    """依赖注入：获取文本嵌入器实例（进程级单例）"""
    settings: Settings = get_settings()
    return TextEmbedder(
        model_name=settings.embedding_model,
        model_dir=settings.embedding_model_dir,
    )


@lru_cache()
def get_vector_store() -> VectorStore:
    """依赖注入：获取向量存储实例（进程级单例）"""
    settings: Settings = get_settings()
    return VectorStore(host=settings.chroma_host, port=settings.chroma_port)


@lru_cache()
def get_document_chunker() -> DocumentChunker:
    """依赖注入：获取文档分块器实例"""
    return DocumentChunker()


def get_document_service(
        db: Session = Depends(get_session),
        embedder: TextEmbedder = Depends(get_embedder),
        vector_store: VectorStore = Depends(get_vector_store),
        chunker: DocumentChunker = Depends(get_document_chunker),
) -> DocumentService:
    """依赖注入：获取文档服务"""
    return DocumentService(
        db=db,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )


def get_tool_registry(
        vector_store: VectorStore = Depends(get_vector_store),
        embedder: TextEmbedder = Depends(get_embedder),
) -> ToolRegistry:
    """依赖注入：获取工具注册表"""
    registry = ToolRegistry()
    vector_search_tool = VectorSearchTool(vector_store, embedder)
    registry.register(vector_search_tool)
    return registry


def get_agent_factory(
        llm: BaseLLM = Depends(get_llm),
        tool_registry: ToolRegistry = Depends(get_tool_registry),
        settings: Settings = Depends(get_settings),
) -> AgentFactory:
    """依赖注入：获取 Agent 工厂实例"""
    return AgentFactory(llm=llm, tool_registry=tool_registry, settings=settings)


def get_runtime_service(factory: AgentFactory = Depends(get_agent_factory), ) -> AgentRuntimeService:
    """依赖注入：获取 Agent 运行时服务"""
    return AgentRuntimeService(agent_factory=factory)


def get_conversation_service(
        db: Session = Depends(get_session),
) -> ConversationService:
    """依赖注入：获取会话服务"""
    return ConversationService(db=db)


def get_chat_service(
        convo_service: ConversationService = Depends(get_conversation_service),
        runtime_service: AgentRuntimeService = Depends(get_runtime_service),
) -> ChatApplicationService:
    """依赖注入：获取聊天应用服务"""
    return ChatApplicationService(
        convo_service=convo_service,
        runtime_service=runtime_service,
    )
