from __future__ import annotations

from langchain_core.messages import HumanMessage

from ..agent.state import AgentState
from ..api.models import ChatRequest, ChatResponse
from sqlmodel import Session as DbSession
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..services.conversation_service import ConversationService
from ..services.runtime_service import AgentRuntimeService
from ..tools.base import ToolRegistry
from ..tools.search import VectorSearchTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatApplicationService:
    """
    应用服务层，负责编排核心服务以完成特定的业务用例。
    """

    def __init__(
            self,
            convo_service: ConversationService,
            runtime_service: AgentRuntimeService,
            embedder: TextEmbedder,
            vector_store: VectorStore,
    ):
        self._convo_service = convo_service
        self._runtime_service = runtime_service
        self._embedder = embedder
        self._vector_store = vector_store

    def handle_chat_request(
            self, req: ChatRequest, db: DbSession
    ) -> ChatResponse:
        """
        处理一个聊天请求的完整业务流程。
        """
        logger.info(f"开始处理聊天请求 | conversation_uid: {req.conversation_uid}")

        # 1. (会话服务) 加载或创建会话，并获取历史消息
        conversation = self._convo_service.get_or_create(req.conversation_uid)
        history_messages = self._convo_service.load_history_messages(conversation)

        # 2. (应用层) 准备特定用例的工具和状态
        # 这是文档问答用例的核心配置
        search_tool = VectorSearchTool(self._vector_store, self._embedder)
        tool_registry = ToolRegistry()
        tool_registry.register(search_tool)

        history_messages.append(HumanMessage(content=req.message))
        initial_state = AgentState(
            messages=history_messages,
            documents=[],
            next_action="search",
            tool_results={},
            iteration_count=0,
            final_answer=None,
            conversation_uid=req.conversation_uid,
            document_uid=req.document_uid,
        )

        # 3. (运行时服务) 执行 Agent
        result_state = self._runtime_service.run(
            state=initial_state, tool_registry=tool_registry, db=db
        )

        # 4. (会话服务) 持久化新一轮的对话
        answer = result_state.get("final_answer", "Agent did not produce an answer.")
        used_docs = result_state.get("documents", [])
        self._convo_service.add_message(conversation, "user", req.message)
        self._convo_service.add_message(conversation, "assistant", answer)

        logger.info(f"聊天请求处理完成 | conversation_uid: {req.conversation_uid}")

        # 5. 返回结构化的响应数据
        return ChatResponse(
            conversation_uid=req.conversation_uid,
            document_uid=req.document_uid,
            user_message=req.message,
            answer=answer,
            documents=used_docs,
        )
