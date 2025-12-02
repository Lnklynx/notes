from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..agent.state import AgentState
from ..llm.base import BaseLLM, LLMRequest, LLMResponse
from ..services.execution_recorder import ExecutionRecorder
from ..tools.base import ToolRegistry


def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    """将 LangChain 消息对象序列化为字典"""
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "system"  # Or handle other types as needed

    content = message.content
    tool_calls = []

    if isinstance(message, AIMessage) and message.tool_calls:
        tool_calls = [
            {"name": tc["name"], "arguments": tc["args"], "id": tc["id"]}
            for tc in message.tool_calls
        ]

    return {"role": role, "content": content, "tool_calls": tool_calls}


def serialize_llm_request(request: LLMRequest) -> Dict[str, Any]:
    """序列化 LLMRequest 为字典"""
    return {
        "messages": [_serialize_message(m) for m in request.messages],
        "tools": request.tools,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }


def serialize_llm_response(response: LLMResponse) -> Dict[str, Any]:
    """序列化 LLMResponse 为字典"""
    return {
        "content": response.content,
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments, "id": tc.id}
            for tc in response.tool_calls
        ]
        if response.tool_calls
        else [],
    }


def _capture_exception(e: Exception) -> Dict[str, Any]:
    """捕获并格式化异常信息"""
    return {
        "error_type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }


class AuditedLLM(BaseLLM):
    """带审计记录功能的 LLM 代理类"""

    def __init__(
            self,
            llm: BaseLLM,
            recorder: ExecutionRecorder,
            state: AgentState,
            node_name: str,
    ):
        self._llm = llm
        self._recorder = recorder
        self._state = state
        self._node_name = node_name

    def chat(self, request: LLMRequest) -> LLMResponse:
        start_time = datetime.now(timezone.utc)
        llm_input = serialize_llm_request(request)
        status = "success"
        error_info = None
        response = None

        try:
            response = self._llm.chat(request)
            return response
        except Exception as e:
            status = "error"
            error_info = _capture_exception(e)
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            llm_output = serialize_llm_response(response) if response else None

            self._recorder.record_step(
                conversation_uid=self._state["conversation_uid"],
                node_name=self._node_name,
                iteration_count=self._state.get("iteration_count", 0),
                status=status,
                execution_time_ms=execution_time_ms,
                llm_input=llm_input,
                llm_output=llm_output,
                error_info=error_info,
            )

    def stream(self, request: LLMRequest):
        # Auditing for streaming is more complex and not implemented for simplicity.
        return self._llm.stream(request)


class AuditedToolRegistry:
    """带审计记录功能的 ToolRegistry 代理类"""

    def __init__(
            self,
            registry: ToolRegistry,
            recorder: ExecutionRecorder,
            state: AgentState,
            node_name: str,
    ):
        # This proxy holds a reference to the original registry, it is not a registry itself.
        self._registry = registry
        self._recorder = recorder
        self._state = state
        self._node_name = node_name

    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        start_time = datetime.now(timezone.utc)
        tool_input = {"tool_name": tool_name, "arguments": kwargs}
        status = "success"
        error_info = None
        tool_output = None

        try:
            tool_output = self._registry.invoke_tool(tool_name, **kwargs)
            return tool_output
        except Exception as e:
            status = "error"
            error_info = _capture_exception(e)
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Ensure output is JSON serializable
            output_to_log = tool_output
            if not isinstance(output_to_log, (dict, list, str, int, float, bool, type(None))):
                output_to_log = str(output_to_log)

            self._recorder.record_step(
                conversation_uid=self._state["conversation_uid"],
                node_name=self._node_name,
                iteration_count=self._state.get("iteration_count", 0),
                status=status,
                execution_time_ms=execution_time_ms,
                tool_input=tool_input,
                tool_output={"result": output_to_log},
                error_info=error_info,
            )

    def get_tool_schemas(self) -> list[dict]:
        return self._registry.get_tool_schemas()

    def get(self, tool_name: str):
        return self._registry.get(tool_name)

    @property
    def tools(self):
        return self._registry.tools
