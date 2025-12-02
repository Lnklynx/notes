from typing import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Agent 状态定义 - LangGraph 状态结构"""

    messages: list[BaseMessage]  # 对话历史
    conversation_uid: str  # 会话唯一标识
    document_uid: str  # 当前文档唯一标识
