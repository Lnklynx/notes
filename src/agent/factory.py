from __future__ import annotations

from langgraph.graph import StateGraph

from ..agent.auditing import AuditedLLM, AuditedToolRegistry
from ..agent.graph import create_agent_graph
from ..agent.state import AgentState
from ..config import Settings
from sqlmodel import Session as DbSession
from ..llm.base import BaseLLM
from ..services.execution_recorder import ExecutionRecorder
from ..tools.base import ToolRegistry


class AuditableGraphFactory:
    """
    一个工厂类，负责创建、审计和组装 Agent Graph。
    它封装了图构建过程中的所有复杂性，使得服务层代码更简洁。
    """

    def __init__(self, llm: BaseLLM, tool_registry: ToolRegistry, settings: Settings):
        self._llm = llm
        self._tool_registry = tool_registry
        self._settings = settings

    def create_graph(self, state: AgentState, db: DbSession) -> StateGraph:
        """
        创建并编译一个带审计功能的 Agent Graph。

        Args:
            state: Agent 的初始状态，包含了会话信息和历史消息。
            db: 数据库会话实例。

        Returns:
            编译好的图实例。
        """
        # 1. 记录器现在在工厂内部创建
        recorder = ExecutionRecorder(session=db)

        # 2. 代理对象的创建被封装在此处
        traced_llm_think = AuditedLLM(
            llm=self._llm, recorder=recorder, state=state, node_name="think"
        )
        traced_llm_synthesize = AuditedLLM(
            llm=self._llm, recorder=recorder, state=state, node_name="synthesize"
        )
        traced_tool_registry = AuditedToolRegistry(
            registry=self._tool_registry,
            recorder=recorder,
            state=state,
            node_name="search",
        )

        # 3. 图的创建和编译逻辑也在这里
        graph_builder = create_agent_graph(
            llm_think=traced_llm_think,
            llm_synthesize=traced_llm_synthesize,
            tool_registry=traced_tool_registry,
            max_iterations=self._settings.max_iterations,
        )

        return graph_builder.compile()
