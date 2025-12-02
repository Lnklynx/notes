from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage
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
    ) -> Message:
        """向会话中添加一条新消息"""
        message = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
        )
        self._db.add(message)
        logger.debug(
            f"[ConversationService] 添加消息到会话 {conversation.id} | role: {role}"
        )
        return message

    def load_history_messages(
            self, conversation: Conversation) -> list[HumanMessage | AIMessage]:
        """从数据库加载历史消息并转换为 LangChain 消息对象"""
        rows: Sequence[Message] = self._db.exec(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                Message.is_deleted == False,  # noqa: E712
            )
            .order_by(Message.created_at)
        ).all()

        history: list[HumanMessage | AIMessage] = []
        for row in rows:
            if row.role == "user":
                history.append(HumanMessage(content=row.content))
            elif row.role == "assistant":
                history.append(AIMessage(content=row.content))
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
