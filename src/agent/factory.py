from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from .graph import create_agent_graph
from ..config import Settings
from ..llm.base import BaseLLM
from ..tools.base import ToolRegistry


class AgentFactory:
    """
    一个工厂类，负责创建 Agent Graph。
    """

    def __init__(self, llm: BaseLLM, tool_registry: ToolRegistry, settings: Settings):
        self._llm = llm
        self._tool_registry = tool_registry
        self._settings = settings
        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """构建并编译图"""
        graph_builder = create_agent_graph(
            llm=self._llm,
            tool_registry=self._tool_registry,
        )
        return graph_builder.compile()

    def get_graph(self) -> CompiledStateGraph:
        """获取编译好的图实例"""
        return self._graph
