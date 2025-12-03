from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from .context import AgentContext


class AgentState(TypedDict, total=False):
    """Agent 状态定义 - LangGraph 状态结构"""
    conversation_uid: str  # 会话唯一标识
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史，使用 add_messages reducer 确保消息被追加
    context: AgentContext | None  # Agent 的结构化工作上下文
