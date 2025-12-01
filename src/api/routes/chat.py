from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json
from ..models import ChatRequest, ChatResponse
from ...llm.factory import create_llm
from ...embedding.embedder import TextEmbedder
from ...embedding.vector_store import VectorStore
from ...tools.base import ToolRegistry
from ...tools.search import VectorSearchTool
from ...agent.graph import create_agent_graph
from ...agent.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from ...config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])

# 会话存储（实际应使用数据库）
sessions: dict[str, dict] = {}


def get_agent_services():
    """获取 Agent 服务"""
    settings = get_settings()
    
    # 根据提供商选择对应的 API 密钥
    provider = settings.llm_provider.lower()
    api_key = ""
    if provider == "openai":
        api_key = settings.openai_api_key
    elif provider == "qwen":
        api_key = settings.qwen_api_key
    
    llm = create_llm(
        provider,
        api_key=api_key,
        model=settings.llm_model,
        base_url=settings.ollama_base_url
    )
    embedder = TextEmbedder(settings.embedding_model)
    vector_store = VectorStore(settings.vector_db_path)
    vector_store.create_collection("documents")
    
    return {"llm": llm, "embedder": embedder, "vector_store": vector_store}


@router.post("/send", response_model=ChatResponse)
async def send_message(
    req: ChatRequest,
    services: dict = Depends(get_agent_services)
):
    """发送消息，获取 AI 响应"""
    try:
        settings = get_settings()
        
        # 初始化会话
        if req.session_id not in sessions:
            sessions[req.session_id] = {"messages": []}
        
        session = sessions[req.session_id]
        
        # 构建工具
        tool_registry = ToolRegistry()
        search_tool = VectorSearchTool(
            services["vector_store"],
            services["embedder"]
        )
        tool_registry.register(search_tool)
        tools_info = tool_registry.list_tools()
        
        # 创建 Agent 图
        graph = create_agent_graph(
            services["llm"],
            tool_registry,
            tools_info,
            max_iterations=settings.max_iterations
        )
        
        # 初始化状态
        session["messages"].append(HumanMessage(content=req.message))
        
        state: AgentState = {
            "messages": session["messages"],
            "documents": [],
            "next_action": "search",
            "tool_results": {},
            "iteration_count": 0,
            "final_answer": None,
            "session_id": req.session_id,
            "document_id": req.document_id
        }
        
        # 执行 Agent
        result = graph.invoke(state)
        
        answer = result.get("final_answer", "No answer generated")
        session["messages"].append(AIMessage(content=answer))
        
        return ChatResponse(
            session_id=req.session_id,
            document_id=req.document_id,
            user_message=req.message,
            answer=answer,
            documents=result.get("documents", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """获取会话历史"""
    if session_id not in sessions:
        return {"session_id": session_id, "messages": []}
    
    messages = []
    for msg in sessions[session_id]["messages"]:
        messages.append({
            "role": "user" if hasattr(msg, '__class__') and msg.__class__.__name__ == "HumanMessage" else "assistant",
            "content": msg.content
        })
    
    return {"session_id": session_id, "messages": messages}

