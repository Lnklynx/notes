import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_chat_service, get_conversation_service
from ..models import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
)
from ..responses import StandardResponse, success_response
from ...services.chat_service import ChatApplicationService
from ...services.conversation_service import ConversationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/completion",
    response_model=StandardResponse[ChatResponse],
    summary="调用 Agent 进行聊天",
)
async def chat_completion(
        req: ChatRequest,
        app_service: ChatApplicationService = Depends(get_chat_service),
) -> StandardResponse:
    """
    调用 Agent 进行聊天。
    此端点现在通过依赖注入获取应用服务，并调用它来处理业务逻辑。
    """
    response_data = app_service.handle_chat_request(req=req)
    return success_response(data=response_data)


@router.get(
    "/history/{conversation_uid}",
    response_model=StandardResponse[HistoryResponse],
    summary="获取会話历史",
)
async def get_history(
        conversation_uid: str,
        convo_service: ConversationService = Depends(get_conversation_service),
) -> StandardResponse:
    """获取会话历史"""
    messages = convo_service.load_history_for_api(conversation_uid)
    response_data = HistoryResponse(
        conversation_uid=conversation_uid, messages=messages
    )
    return success_response(data=response_data)
