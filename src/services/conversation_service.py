from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from sqlmodel import Session, select

from ..db.models import Conversation, Message
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConversationService:
    """
    纯粹的会话持久化和管理服务。
    负责所有与 Conversation 和 Message 模型相关的数据库操作。
    """

    def __init__(self, db: Session):
        self._db = db

    def get_or_create(self, conversation_uid: str) -> Conversation:
        """根据 conversation_uid 获取或创建会话"""
        conversation = self._db.exec(
            select(Conversation).where(
                Conversation.conversation_uid == conversation_uid,
                Conversation.is_deleted == False,  # noqa: E712
            )
        ).first()

        if conversation is None:
            logger.info(f"[ConversationService] 创建新会话: {conversation_uid}")
            conversation = Conversation(
                conversation_uid=conversation_uid,
            )
            self._db.add(conversation)
            self._db.flush()  # flush to get the conversation.id for messages

        return conversation

    def add_message(
            self,
            conversation: Conversation,
            role: str,
            content: str,
            extra: dict | None = None,
    ) -> Message:
        """向会话中添加一条新消息"""
        message = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            extra=extra,
        )
        self._db.add(message)
        logger.debug(
            f"[ConversationService] 添加消息到会话 {conversation.id} | role: {role}"
        )
        return message

    def add_langchain_message(
            self,
            conversation: Conversation,
            langchain_message: BaseMessage,
    ) -> Message:
        """保存 LangChain 消息对象到数据库"""
        if isinstance(langchain_message, HumanMessage):
            role = "user"
            content = langchain_message.content
            extra = None
        elif isinstance(langchain_message, AIMessage):
            role = "assistant"
            content = langchain_message.content
            # 保存工具调用信息到 extra 字段
            extra = {}
            if hasattr(langchain_message, "tool_calls") and langchain_message.tool_calls:
                extra["tool_calls"] = [
                    {
                        "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""),
                        "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "arguments", getattr(tc, "args", {})),
                        "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", ""),
                    }
                    for tc in langchain_message.tool_calls
                ]
            extra = extra if extra else None
        elif isinstance(langchain_message, ToolMessage):
            role = "tool"
            content = langchain_message.content
            # 保存工具调用ID到 extra 字段
            extra = {
                "tool_call_id": langchain_message.tool_call_id,
            }
        else:
            # 其他类型的消息（如 SystemMessage）
            role = getattr(langchain_message, "type", "system")
            content = langchain_message.content
            extra = None

        return self.add_message(conversation, role, content, extra)

    def load_history_messages(
            self, conversation: Conversation) -> list[BaseMessage]:
        """从数据库加载历史消息并转换为 LangChain 消息对象
        
        注意：确保 ToolMessage 必须紧跟在包含 tool_calls 的 AIMessage 之后。
        如果历史数据中有孤立的 ToolMessage（前面的 AIMessage 没有 tool_calls），
        会跳过它以避免 LLM API 报错。
        """
        rows: Sequence[Message] = self._db.exec(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                Message.is_deleted == False,  # noqa: E712
            )
            .order_by(Message.created_at)
        ).all()

        history: list[BaseMessage] = []
        prev_assistant_has_tool_calls = False
        
        for row in rows:
            if row.role == "user":
                history.append(HumanMessage(content=row.content))
                prev_assistant_has_tool_calls = False
            elif row.role == "assistant":
                # 恢复工具调用信息
                tool_calls = None
                if row.extra and "tool_calls" in row.extra:
                    tool_calls = row.extra["tool_calls"]
                ai_message = AIMessage(content=row.content)
                if tool_calls:
                    ai_message.tool_calls = tool_calls
                    prev_assistant_has_tool_calls = True
                else:
                    prev_assistant_has_tool_calls = False
                history.append(ai_message)
            elif row.role == "tool":
                # 恢复工具消息
                # 确保 ToolMessage 必须紧跟在包含 tool_calls 的 AIMessage 之后
                if not prev_assistant_has_tool_calls:
                    # 跳过孤立的 ToolMessage（可能是旧数据或数据损坏）
                    logger.warning(
                        f"跳过孤立的 ToolMessage（前面的 AIMessage 没有 tool_calls）| "
                        f"conversation_id: {conversation.id}, message_id: {row.id}"
                    )
                    continue
                
                tool_call_id = None
                if row.extra and "tool_call_id" in row.extra:
                    tool_call_id = row.extra["tool_call_id"]
                history.append(ToolMessage(content=row.content, tool_call_id=tool_call_id or ""))
                # ToolMessage 后，重置标志（下一个 ToolMessage 需要新的 assistant 消息）
                prev_assistant_has_tool_calls = False
            # 其他类型的消息可以在这里扩展
        return history

    def load_history_for_api(self, conversation_uid: str) -> list[dict]:
        """提供给 API 的历史消息数据结构"""
        conversation = self._db.exec(
            select(Conversation).where(
                Conversation.conversation_uid == conversation_uid,
                Conversation.is_deleted == False,  # noqa: E712
            )
        ).first()

        if conversation is None:
            return []

        rows: Sequence[Message] = self._db.exec(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                Message.is_deleted == False,  # noqa: E712
            )
            .order_by(Message.created_at)
        ).all()

        return [{"role": row.role, "content": row.content} for row in rows]
