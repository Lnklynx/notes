from __future__ import annotations

from langchain_core.messages import HumanMessage

from ..agent.state import AgentState
from ..api.models import ChatRequest, ChatResponse
from ..services.conversation_service import ConversationService
from ..services.runtime_service import AgentRuntimeService
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
    ):
        self._convo_service = convo_service
        self._runtime_service = runtime_service

    def handle_chat_request(self, req: ChatRequest) -> ChatResponse:
        """
        处理一个聊天请求的完整业务流程。
        """
        logger.info(f"开始处理聊天请求 | conversation_uid: {req.conversation_uid}")

        # 1. (会话服务) 加载或创建会话，并获取历史消息
        conversation = self._convo_service.get_or_create(req.conversation_uid)
        history_messages = self._convo_service.load_history_messages(conversation)
        history_messages.append(HumanMessage(content=req.message))

        # 2. (应用层) 准备 Agent 初始状态
        initial_state = AgentState(
            messages=history_messages,
            document_uid=req.document_uid,
            conversation_uid=req.conversation_uid,
        )

        # 3. (运行时服务) 执行 Agent
        result_state = self._runtime_service.run(state=initial_state)

        # 4. (会话服务) 持久化新一轮的对话
        answer = result_state["messages"][-1].content
        self._convo_service.add_message(conversation, "user", req.message)
        self._convo_service.add_message(conversation, "assistant", answer)

        logger.info(f"聊天请求处理完成 | conversation_uid: {req.conversation_uid}")

        # 5. 返回结构化的响应数据
        return ChatResponse(
            conversation_uid=req.conversation_uid,
            document_uid=req.document_uid,
            user_message=req.message,
            answer=answer,
            documents=[],
        )
