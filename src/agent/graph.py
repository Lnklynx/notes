from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import llm_node, tool_node
from ..llm.base import BaseLLM
from ..tools.base import ToolRegistry


def should_continue(state: AgentState) -> str:
    """判断是否继续执行工具"""
    last_message = state['messages'][-1]
    # 如果没有工具调用，则结束
    if not last_message.tool_calls:
        return "end"
    # 否则，继续执行工具
    return "continue"


def create_agent_graph(llm: BaseLLM, tool_registry: ToolRegistry):
    """创建 Agent 图"""

    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("agent", lambda state: llm_node(state, llm, tool_registry))
    graph.add_node("action", lambda state: tool_node(state, tool_registry))

    # 定义边
    graph.set_entry_point("agent")

    # 条件边：根据 agent 的输出决定是调用工具还是结束
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # 从 action 返回到 agent
    graph.add_edge("action", "agent")

    return graph
