from __future__ import annotations

from typing import Sequence

from sqlmodel import Session, select
from langchain_core.messages import HumanMessage, AIMessage

from ..config import Settings
from ..db.models import Conversation, Message
from ..agent.state import AgentState
from ..agent.graph import create_agent_graph
from ..tools.base import ToolRegistry
from ..tools.search import VectorSearchTool
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore
from ..llm.base import BaseLLM


def ensure_conversation(
        db: Session,
        conversation_uid: str,
) -> Conversation:
    """根据 conversation_uid 获取或创建会话"""
    conversation = db.exec(
        select(Conversation).where(
            Conversation.conversation_uid == conversation_uid,
            Conversation.is_deleted == False,  # noqa: E712
        )
    ).first()

    if conversation is None:
        conversation = Conversation(
            conversation_uid=conversation_uid,
            title=None,
            user_id=None,
        )
        db.add(conversation)
        db.flush()

    return conversation


def load_history_messages(db: Session, conversation: Conversation) -> list[HumanMessage | AIMessage]:
    """从数据库加载历史消息并转换为 LangChain 消息"""
    rows: Sequence[Message] = db.exec(
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


def run_conversation_turn(
        *,
        db: Session,
        settings: Settings,
        conversation_uid: str,
        document_uid: str,
        user_message: str,
        llm: BaseLLM,
        embedder: TextEmbedder,
        vector_store: VectorStore,
) -> tuple[str, list[str]]:
    """执行单轮对话：加载历史 → 调 Agent → 持久化消息"""

    # 1. 会话与历史
    conversation = ensure_conversation(db, conversation_uid)
    history_messages = load_history_messages(db, conversation)

    # 2. 构建工具与 Agent
    tool_registry = ToolRegistry()
    search_tool = VectorSearchTool(vector_store, embedder)
    tool_registry.register(search_tool)
    tools_info = tool_registry.list_tools()

    graph = create_agent_graph(
        llm,
        tool_registry,
        tools_info,
        max_iterations=settings.max_iterations,
    )

    # 3. 构造 Agent 状态
    history_messages.append(HumanMessage(content=user_message))
    state: AgentState = {
        "messages": history_messages,
        "documents": [],
        "next_action": "search",
        "tool_results": {},
        "iteration_count": 0,
        "final_answer": None,
        "conversation_uid": conversation_uid,
        "document_uid": document_uid,
    }

    # 4. 执行 Agent
    result = graph.invoke(state)
    answer = result.get("final_answer", "No answer generated")
    used_docs: list[str] = result.get("documents", [])

    # 5. 持久化当前轮对话
    db.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
        )
    )
    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
        )
    )

    return answer, used_docs


def load_history_for_api(db: Session, conversation_uid: str) -> list[dict]:
    """提供给 API 的历史消息数据结构"""
    conversation = db.exec(
        select(Conversation).where(
            Conversation.conversation_uid == conversation_uid,
            Conversation.is_deleted == False,  # noqa: E712
        )
    ).first()

    if conversation is None:
        return []

    rows: Sequence[Message] = db.exec(
        select(Message)
        .where(
            Message.conversation_id == conversation.id,
            Message.is_deleted == False,  # noqa: E712
        )
        .order_by(Message.created_at)
    ).all()

    return [
        {
            "role": row.role,
            "content": row.content,
        }
        for row in rows
    ]
