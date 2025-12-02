from .state import AgentState
from ..llm.base import BaseLLM, LLMResponse, LLMRequest
from ..tools.base import ToolRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


def think_node(
    state: AgentState,
    llm: BaseLLM,
    tool_registry: ToolRegistry,
    temperature: float = 0.7,
) -> dict:
    """思考节点：分析问题，决定下一步动作（使用结构化工具调用）"""

    last_message = state["messages"][-1].content if state["messages"] else ""
    iteration = state.get("iteration_count", 0)

    logger.info(f"[Think Node] 迭代 {iteration} | 分析问题: {last_message[:100]}...")

    messages = state["messages"].copy()
    tools = tool_registry.get_tool_schemas()

    request = LLMRequest(
        messages=messages, tools=tools if tools else None, temperature=temperature
    )

    response: LLMResponse = llm.chat(request)

    next_action = "finish"
    tool_calls_info = []

    if response.has_tool_calls():
        for tool_call in response.tool_calls:
            tool_name = tool_call.name
            tool_calls_info.append({"name": tool_name, "arguments": tool_call.arguments})

            if tool_name == "vector_search":
                next_action = "search"
                logger.info(
                    f"[Think Node] LLM 请求调用工具: {tool_name} | 参数: {tool_call.arguments}"
                )
                break
    else:
        logger.info(f"[Think Node] LLM 未请求工具调用 | 响应: {response.content[:200]}...")

    return {
        "next_action": next_action,
        "tool_results": {
            "think_response": response.content,
            "tool_calls": tool_calls_info,
        },
    }


def search_node(state: AgentState, tool_registry: ToolRegistry, top_k: int = 5) -> dict:
    """搜索节点：调用检索工具"""

    last_message = state["messages"][-1].content if state["messages"] else ""
    document_uid = state.get("document_uid")

    tool_results = state.get("tool_results", {})
    tool_calls = tool_results.get("tool_calls", [])

    query = last_message
    search_top_k = top_k
    search_doc_uid = document_uid

    for tool_call in tool_calls:
        if tool_call.get("name") == "vector_search":
            args = tool_call.get("arguments", {})
            query = args.get("query", query)
            search_top_k = args.get("top_k", search_top_k)
            if "document_uid" in args:
                search_doc_uid = args.get("document_uid", search_doc_uid)
            break

    logger.info(f"[Search Node] 执行检索 | query: {query[:100]}... | top_k: {search_top_k} | document_uid: {search_doc_uid}")

    tool_input = {
        "query": query,
        "top_k": search_top_k,
        "document_uid": search_doc_uid,
    }

    results = tool_registry.invoke_tool("vector_search", **tool_input)

    documents = results.get("documents", [[]])[0]
    doc_count = len(documents) if documents else 0

    logger.info(f"[Search Node] 检索完成 | 找到 {doc_count} 个相关文档片段")
    if doc_count == 0:
        logger.warning(f"[Search Node] ⚠️ 未找到相关文档！可能原因：1) Chroma 中无数据 2) document_uid 过滤后无结果 3) 查询与文档不匹配")
    else:
        logger.debug(f"[Search Node] 文档片段预览: {documents[0][:150] if documents else 'N/A'}...")

    return {
        "documents": documents,
        "tool_results": {"search_results": results},
    }


def synthesize_node(state: AgentState, llm: BaseLLM, max_context: int = 3000, temperature: float = 0.7) -> dict:
    """合成节点：结合检索结果生成回答"""

    documents = state.get("documents", [])
    doc_count = len(documents)
    last_message = state["messages"][-1].content if state["messages"] else ""

    logger.info(f"[Synthesize Node] 生成回答 | 文档片段数: {doc_count} | 问题: {last_message[:100]}...")

    if not documents or all(not doc.strip() for doc in documents):
        logger.warning(f"[Synthesize Node] ⚠️ 没有可用的文档内容，将告知用户")
        answer = "抱歉，我无法在提供的文档中找到相关信息。请确认：1) 文档已正确上传并向量化 2) 问题与文档内容相关 3) 文档在 Chroma 向量库中可检索到。"
    else:
        context = "\n\n".join(documents[:5])[:max_context]
        logger.debug(f"[Synthesize Node] 上下文长度: {len(context)} 字符")

        prompt = f"""基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{last_message}

请提供详细且准确的回答。如果文档内容不足以回答问题，请如实说明。"""

        # Note: We are creating a new list, not modifying the one in the state
        messages = list(state["messages"]) + [{"role": "user", "content": prompt}]

        request = LLMRequest(
            messages=messages,
            temperature=temperature
        )
        response: LLMResponse = llm.chat(request)
        answer = response.content
        logger.info(f"[Synthesize Node] 回答生成完成 | 长度: {len(answer)} 字符")

    return {
        "final_answer": answer,
        "next_action": "finish"
    }


def judge_node(state: AgentState, max_iterations: int = 10) -> dict:
    """判断节点：决定是否继续迭代或结束"""

    current_iter = state.get("iteration_count", 0)
    next_iter = current_iter + 1

    logger.info(f"[Judge Node] 迭代 {current_iter} -> {next_iter} | 最大迭代: {max_iterations}")

    if current_iter >= max_iterations:
        logger.warning(f"[Judge Node] 达到最大迭代次数，强制结束")
        return {"next_action": "finish", "iteration_count": next_iter}

    next_action = state.get("next_action", "finish")
    logger.info(f"[Judge Node] 下一步动作: {next_action}")

    return {"next_action": next_action, "iteration_count": next_iter}
