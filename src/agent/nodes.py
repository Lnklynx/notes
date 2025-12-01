from .state import AgentState
from ..llm.base import BaseLLM
from ..tools.base import ToolRegistry
import json


def think_node(state: AgentState, llm: BaseLLM, tools_info: list[dict]) -> dict:
    """思考节点：分析问题，决定下一步动作"""

    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = f"""根据用户问题进行分析：
    
问题：{last_message}

可用工具：
{json.dumps(tools_info, ensure_ascii=False, indent=2)}

请判断是否需要检索文档。回答格式：
ACTION: search 或 finish
REASON: 简要说明"""

    messages = state["messages"] + [{"role": "user", "content": prompt}]
    response = llm.chat(messages)

    next_action = "search" if "search" in response.lower() else "finish"

    return {
        "next_action": next_action,
        "tool_results": {"think_response": response}
    }


def search_node(state: AgentState, tool_registry: ToolRegistry, top_k: int = 5) -> dict:
    """搜索节点：调用检索工具"""

    last_message = state["messages"][-1].content if state["messages"] else ""
    search_tool = tool_registry.get("vector_search")

    # 按当前对话绑定的文档进行过滤检索
    document_uid = state.get("document_uid")

    results = search_tool.execute(
        query=last_message,
        top_k=top_k,
        document_uid=document_uid,
    )

    documents = results.get("documents", [[]])[0]

    return {
        "documents": documents,
        "tool_results": {"search_results": results},
    }


def synthesize_node(state: AgentState, llm: BaseLLM, max_context: int = 3000) -> dict:
    """合成节点：结合检索结果生成回答"""

    context = "\n\n".join(state["documents"][:5])[:max_context]
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = f"""基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{last_message}

请提供详细且准确的回答。"""

    messages = state["messages"] + [{"role": "user", "content": prompt}]
    answer = llm.chat(messages)

    return {
        "final_answer": answer,
        "next_action": "finish"
    }


def judge_node(state: AgentState, max_iterations: int = 10) -> dict:
    """判断节点：决定是否继续迭代或结束"""

    if state["iteration_count"] >= max_iterations:
        return {"next_action": "finish"}

    return {"next_action": state.get("next_action", "finish")}
