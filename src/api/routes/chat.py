import logging

from fastapi import APIRouter, Depends

from ..dependencies import LLMComponents, build_llm_components
from ..models import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
)
from ..responses import StandardResponse, success_response
from ...config import get_settings
from ...db.session import get_session
from ...services.chat_service import ChatApplicationService
from ...services.conversation_service import ConversationService
from ...services.runtime_service import AgentRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def get_llm_services() -> LLMComponents:
    """依赖注入：获取 LLM 相关服务"""
    settings = get_settings()
    # TODO: 后续 embedder 和 vector_store 也应通过依赖注入管理
    components = build_llm_components(settings)
    return components


@router.post(
    "/completion",
    response_model=StandardResponse[ChatResponse],
    summary="调用 Agent 进行聊天",
)
async def invoke_agent(
        req: ChatRequest, llm_components: LLMComponents = Depends(get_llm_services)
) -> StandardResponse:
    """
    调用 Agent 进行聊天。
    此端点保持轻量，仅负责 HTTP 接口转换和调用应用服务。
    """
    settings = get_settings()
    with get_session() as db:
        # 1. 初始化所有需要的服务
        convo_service = ConversationService(db=db)
        runtime_service = AgentRuntimeService(
            llm=llm_components["llm"], settings=settings
        )
        app_service = ChatApplicationService(
            convo_service=convo_service,
            runtime_service=runtime_service,
            embedder=llm_components["embedder"],
            vector_store=llm_components["vector_store"],
        )
        # 2. 调用应用服务处理业务逻辑
        response_data = app_service.handle_chat_request(req=req, db=db)
        return success_response(data=response_data)


@router.get(
    "/history/{conversation_uid}",
    response_model=StandardResponse[HistoryResponse],
    summary="获取会话历史",
)
async def get_history(
        conversation_uid: str,
) -> StandardResponse:
    """获取会话历史"""
    with get_session() as db:
        convo_service = ConversationService(db=db)
        messages = convo_service.load_history_for_api(conversation_uid)
        response_data = HistoryResponse(
            conversation_uid=conversation_uid, messages=messages
        )
        return success_response(data=response_data)
