from typing import TypedDict, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class AgentState(TypedDict):
    """Agent 状态定义 - LangGraph 状态结构"""
    
    messages: list[BaseMessage]           # 对话历史
    documents: list[str]                  # 检索到的文档片段
    next_action: str                      # 下一步行动（search/synthesize/finish）
    tool_results: dict[str, Any]         # 工具执行结果
    iteration_count: int                  # 当前迭代次数
    final_answer: Optional[str]          # 最终答案
    conversation_uid: str                 # 会话唯一标识
    document_uid: str                     # 当前文档唯一标识

