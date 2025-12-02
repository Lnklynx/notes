from __future__ import annotations

from typing import Any, Dict, Optional
from sqlmodel import Session, select

from ..db.models import ExecutionRecord
from ..db.session import get_session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionRecorder:
    """执行链路记录器"""

    def __init__(self, session: Optional[Session] = None):
        self._session = session
        self._should_close_session = session is None

    def _get_db_session(self) -> Session:
        """获取数据库会话"""
        if self._session is None:
            self._session = get_session()
        return self._session

    def record_step(
        self,
        conversation_uid: str,
        node_name: str,
        iteration_count: int = 0,
        status: str = "success",
        execution_time_ms: int = 0,
        llm_input: Optional[Dict[str, Any]] = None,
        llm_output: Optional[Dict[str, Any]] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> ExecutionRecord:
        """记录一个执行步骤

        Args:
            conversation_uid: 对话ID
            node_name: 节点名称
            iteration_count: 迭代次数
            status: 执行状态
            execution_time_ms: 执行耗时（毫秒）
            llm_input: LLM 完整输入
            llm_output: LLM 完整输出
            tool_input: 工具输入
            tool_output: 工具输出
            error_info: 错误信息

        Returns:
            ExecutionRecord: 创建的记录
        """
        record = ExecutionRecord(
            conversation_uid=conversation_uid,
            node_name=node_name,
            iteration_count=iteration_count,
            status=status,
            execution_time_ms=execution_time_ms,
            llm_input=llm_input,
            llm_output=llm_output,
            tool_input=tool_input,
            tool_output=tool_output,
            error_info=error_info,
        )

        session = self._get_db_session()
        session.add(record)
        session.commit()
        session.refresh(record)

        logger.debug(
            f"[ExecutionRecorder] 记录步骤 | record_uid: {record.record_uid} | "
            f"node: {node_name} | status: {status}"
        )

        return record

    def close(self):
        """如果需要，关闭数据库会话"""
        if self._session and self._should_close_session:
            self._session.close()
            self._session = None


def get_records_by_conversation(
    conversation_uid: str, session: Optional[Session] = None
) -> list[ExecutionRecord]:
    """按会话查询执行记录

    Args:
        conversation_uid: 会话唯一标识
        session: 数据库会话，如果不提供则创建临时会话

    Returns:
        list[ExecutionRecord]: 执行记录列表，按创建时间排序
    """

    def _get_records(db_session: Session) -> list[ExecutionRecord]:
        return db_session.exec(
            select(ExecutionRecord)
            .where(ExecutionRecord.conversation_uid == conversation_uid)
            .order_by(ExecutionRecord.created_at)
        ).all()

    if session:
        return _get_records(session)
    else:
        with get_session() as db:
            return _get_records(db)

