from __future__ import annotations
from ..agent.factory import AgentFactory
from ..agent.state import AgentState
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentRuntimeService:
    """
    一个封装和执行 Agent 核心逻辑的服务。
    它负责管理 Agent Graph 的生命周期，处理状态流转。
    """

    def __init__(self, agent_factory: AgentFactory):
        self._agent_factory = agent_factory

    def run(self, state: AgentState) -> AgentState:
        """ 执行一个通用的 Agent 流程。"""
        logger.info(
            f"[AgentRuntime] 开始执行 Agent | conversation_uid: {state.get('conversation_uid')}"
        )

        graph = self._agent_factory.get_graph()
        config = {"configurable": {"thread_id": state.get("conversation_uid")}}

        # 执行并返回最终状态
        final_state = graph.invoke(state, config=config)

        answer_length = len(final_state["messages"][-1].content)
        logger.info(
            f"[AgentRuntime] Agent 执行完成 | 回答长度: {answer_length}"
        )

        return final_state
