from __future__ import annotations

from sqlmodel import Session as DbSession
from ..agent.factory import AuditableGraphFactory
from ..agent.state import AgentState
from ..config import Settings
from ..llm.base import BaseLLM
from ..tools.base import ToolRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentRuntimeService:
    """
    一个封装和执行 Agent 核心逻辑的服务。
    它负责管理 Agent Graph 的生命周期，处理状态流转，
    并且是无状态和通用的。
    """

    def __init__(self, llm: BaseLLM, settings: Settings):
        self._llm = llm
        self._settings = settings

    def run(
            self, state: AgentState, tool_registry: ToolRegistry, db: DbSession
    ) -> AgentState:
        """
        执行一个通用的 Agent 流程。

        它接收所有必要的上下文（状态、工具），执行图，并返回最终状态。

        Args:
            state: Agent 的初始状态。
            tool_registry: 为此次运行提供的工具注册表。
            db: 数据库会话，用于审计记录。

        Returns:
            Agent 执行完成后的最终状态。
        """
        logger.info(
            f"[AgentRuntime] 开始执行 Agent | conversation_uid: {state.get('conversation_uid')}"
        )

        graph_factory = AuditableGraphFactory(
            llm=self._llm, tool_registry=tool_registry, settings=self._settings
        )
        graph = graph_factory.create_graph(state=state, db=db)

        # 执行并返回最终状态
        final_state = graph.invoke(state)

        iteration_count = final_state.get("iteration_count", 0)
        used_docs_count = len(final_state.get("documents", []))
        answer_length = len(final_state.get("final_answer", ""))

        logger.info(
            f"[AgentRuntime] Agent 执行完成 | 迭代次数: {iteration_count} | "
            f"使用文档片段数: {used_docs_count} | 回答长度: {answer_length}"
        )

        return final_state
