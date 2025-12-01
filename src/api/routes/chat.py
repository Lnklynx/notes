from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select
from ..models import ChatRequest, ChatResponse, HistoryResponse
from ...config import get_settings
from ...db.session import get_session
from ...services.conversation_service import (
    run_conversation_turn,
    load_history_for_api,
)
from ...services.runtime_service import build_llm_components

router = APIRouter(prefix="/chat", tags=["chat"])


def get_agent_services():
    """获取 Agent 服务"""
    settings = get_settings()

    components = build_llm_components(settings)
    # 这里不创建 collection，交由具体业务（如文档上传）决定
    return components


@router.post("/send", response_model=ChatResponse)
async def send_message(
        req: ChatRequest,
        services: dict = Depends(get_agent_services)
):
    """发送消息，获取 AI 响应"""
    try:
        settings = get_settings()

        with get_session() as db:
            answer, documents = run_conversation_turn(
                db=db,
                settings=settings,
                conversation_uid=req.conversation_uid,
                document_uid=req.document_uid,
                user_message=req.message,
                llm=services["llm"],
                embedder=services["embedder"],
                vector_store=services["vector_store"],
            )

            return ChatResponse(
                conversation_uid=req.conversation_uid,
                document_uid=req.document_uid,
                user_message=req.message,
                answer=answer,
                documents=documents,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{conversation_uid}", response_model=HistoryResponse)
async def get_history(conversation_uid: str):
    """获取会话历史（从 PostgreSQL 加载）"""
    with get_session() as db:
        messages = load_history_for_api(db, conversation_uid)
        return HistoryResponse(conversation_uid=conversation_uid, messages=messages)
