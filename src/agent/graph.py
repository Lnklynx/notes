from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import think_node, search_node, synthesize_node, judge_node
from ..llm.base import BaseLLM
from ..tools.base import ToolRegistry


def create_agent_graph(
        llm_think: BaseLLM,
        llm_synthesize: BaseLLM,
        tool_registry: ToolRegistry,
        max_iterations: int = 10,
):
    """创建 ReAct Agent 图"""

    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node(
        "think",
        lambda state: think_node(state, llm_think, tool_registry),
    )
    graph.add_node(
        "search",
        lambda state: search_node(state, tool_registry),
    )
    graph.add_node(
        "synthesize",
        lambda state: synthesize_node(state, llm_synthesize),
    )
    graph.add_node("judge", lambda state: judge_node(state, max_iterations))

    # 定义边（转移条件）
    graph.set_entry_point("think")

    graph.add_conditional_edges(
        "think",
        lambda state: "search" if state.get("next_action") == "search" else "synthesize"
    )

    graph.add_edge("search", "judge")

    graph.add_conditional_edges(
        "judge",
        lambda state: "think" if state.get("next_action") == "search" else "synthesize"
    )

    graph.add_edge("synthesize", END)

    return graph
