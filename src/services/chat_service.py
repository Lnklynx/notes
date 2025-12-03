from __future__ import annotations
import yaml

from langchain_core.messages import HumanMessage, SystemMessage

from ..agent.state import AgentState
from ..agent.context import AgentContext
from ..api.models import ChatRequest, ChatResponse, ResourceScope
from ..prompts import get_system_prompt
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

        # 如果是新会话，添加静态的系统消息
        if not history_messages:
            system_prompt = get_system_prompt("agent")
            history_messages.insert(0, SystemMessage(content=system_prompt))

        # 2. (应用层) 准备 Agent 初始状态 和 动态上下文
        # 如果请求中没有 scope，则创建一个默认的
        scope = req.scope if req.scope is not None else ResourceScope()

        # 创建结构化的上下文对象，用于 Agent 内部状态
        initial_context = AgentContext(scope=scope)

        # 将动态上下文格式化，注入到最新的人类消息中，供 LLM 读取
        # 使用 YAML 格式以获得更好的可读性
        context_str = yaml.dump(
            initial_context.model_dump(exclude_none=True),
            sort_keys=False,
            default_flow_style=False
        )
        human_message_content = f"""<context>
{context_str}
</context>

{req.message}"""

        # 记录添加用户消息之前的消息数量，用于后续识别新产生的消息
        history_count_before_user = len(history_messages)
        
        history_messages.append(HumanMessage(content=human_message_content))

        # 准备 Agent 初始状态
        initial_state = AgentState(
            context=initial_context,
            messages=history_messages,
            conversation_uid=req.conversation_uid
        )

        # 3. (运行时服务) 执行 Agent
        result_state = self._runtime_service.run(state=initial_state)

        # 4. (会话服务) 持久化新一轮的对话
        # 保存所有新产生的消息（包括用户消息、AI消息和工具消息）
        result_messages = result_state["messages"]
        
        # 从用户消息开始，保存所有新产生的消息
        # 注意：用户消息需要保存原始内容（不包含上下文注入）
        for i, msg in enumerate(result_messages[history_count_before_user:], start=history_count_before_user):
            if i == history_count_before_user:
                # 第一条新消息是用户消息，保存原始内容（不包含上下文注入）
                self._convo_service.add_message(conversation, "user", req.message)
            else:
                # 其他消息（AI消息、工具消息等）完整保存
                self._convo_service.add_langchain_message(conversation, msg)
        
        # 获取最终答案（最后一条消息的内容）
        answer = result_state["messages"][-1].content

        logger.info(f"聊天请求处理完成 | conversation_uid: {req.conversation_uid}")

        # 5. 返回结构化的响应数据
        # TODO: 从 result_state.context.retrieved_content 中提取真实的引用文档
        return ChatResponse(
            conversation_uid=req.conversation_uid,
            user_message=req.message,
            answer=answer,
            source_documents=[],
            retrieved_content=[],
        )
