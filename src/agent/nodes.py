import json
from langchain_core.messages import AIMessage, ToolMessage

from .state import AgentState
from ..llm.base import BaseLLM, LLMResponse, LLMRequest
from ..tools.base import ToolRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


def llm_node(state: AgentState, llm: BaseLLM, tool_registry: ToolRegistry, temperature: float = 0.7) -> dict:
    """
    LLM 节点：调用 LLM 生成响应或工具调用。
    """
    logger.info(f"LLM 节点执行中... | 会话 ID: {state.get('conversation_uid')}")

    request = LLMRequest(
        messages=state["messages"],
        tools=tool_registry.get_tool_schemas(),
        temperature=temperature,
    )

    response: LLMResponse = llm.chat(request)
    logger.info("LLM 调用完成。")

    # 不做处理，有graph控制调用逻辑
    if response.has_tool_calls():
        logger.info(f"LLM 请求调用工具: {[call.name for call in response.tool_calls]}")
    else:
        logger.info("LLM 未请求工具调用，直接生成回复。")

    # 将 ToolCall 对象转换为 LangChain 期望的字典格式
    langchain_tool_calls = []
    for tc in response.tool_calls:
        langchain_tool_calls.append({
            "name": tc.name,
            "args": tc.arguments,
            "id": tc.id,
        })

    # 根据 LLM 的响应构建 AIMessage
    ai_message_kwargs = {"content": response.content}
    if langchain_tool_calls:
        ai_message_kwargs["tool_calls"] = langchain_tool_calls
    ai_message = AIMessage(**ai_message_kwargs)

    # LLM产生的message会被LangGraph自动添加到state["messages"]中，并在ChatApplicationService中持久化
    return {"messages": [ai_message]}


def tool_node(state: AgentState, tool_registry: ToolRegistry) -> dict:
    """
    工具执行节点：根据上一条消息中的工具调用，执行对应工具并返回 ToolMessage。
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    results: list[ToolMessage] = []
    for call in tool_calls:
        # 兼容字典格式和对象格式的 tool_calls
        if isinstance(call, dict):
            name = call.get("name", "")
            args = call.get("args", {})
            call_id = call.get("id", f"call_{name}")
        else:
            name = getattr(call, "name", "")
            args = getattr(call, "arguments", getattr(call, "args", {}))
            call_id = getattr(call, "id", f"call_{name}")

        # 确保 args 是字典类型，如果是字符串则解析 JSON
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                logger.warning(f"工具调用参数不是有效的 JSON: {args}")
                args = {}
        elif args is None:
            args = {}
        elif not isinstance(args, dict):
            logger.warning(f"工具调用参数类型不正确，期望 dict，得到 {type(args)}: {args}")
            args = {}

        logger.info(f"执行工具调用: {name} | args={args}")
        try:
            # 记录参数详情以便调试
            logger.debug(f"工具调用参数详情: name={name}, args类型={type(args)}, args内容={args}")
            output = tool_registry.invoke_tool(name, **args)
        except TypeError as e:
            # 参数类型错误，可能是缺少必需参数
            error_msg = str(e)
            logger.error(f"工具调用参数错误: {name} | 错误: {error_msg} | args: {args}")
            output = f"工具调用失败: 参数错误 - {error_msg}"
        except Exception as e:
            logger.error(f"工具调用失败: {name} | 错误: {str(e)} | args: {args}")
            output = f"工具调用失败: {str(e)}"

        results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=call_id,
            )
        )
    # 工具产生的message会被LangGraph自动添加到state["messages"]中，并在ChatApplicationService中持久化
    # 注意：results 已经是列表，直接返回，LangGraph 会自动追加到现有消息列表
    return {"messages": results}
