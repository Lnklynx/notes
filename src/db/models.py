from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class SQLModelBase(SQLModel):
    """基类：所有模型共享的配置

    约定：
    - 所有表都有自增主键 id（仅内部使用）
    - is_deleted 逻辑删除标记
    - created_at / updated_at 时间戳
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    is_deleted: bool = Field(default=False, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="记录创建时间（UTC，带时区）",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="记录更新时间（UTC，带时区）",
    )

    class Config:
        arbitrary_types_allowed = True


class User(SQLModelBase, table=True):
    """用户（预留，当前可选用）"""

    __tablename__ = "notes_user"

    # 对外暴露的稳定标识
    user_uid: str = Field(
        default_factory=lambda: uuid4().hex, index=True, unique=True
    )
    name: Optional[str] = None


class Conversation(SQLModelBase, table=True):
    """会话"""

    __tablename__ = "notes_conversation"

    conversation_uid: str = Field(
        default_factory=lambda: uuid4().hex, index=True, unique=True
    )
    title: Optional[str] = None
    # 业务层维护的关联字段，仅作标记使用
    user_id: Optional[int] = Field(default=None, index=True)


class Message(SQLModelBase, table=True):
    """消息"""

    __tablename__ = "notes_message"

    message_uid: str = Field(
        default_factory=lambda: uuid4().hex, index=True, unique=True
    )
    conversation_id: int = Field(index=True)
    role: str = Field(max_length=20)  # user / assistant / system
    content: str
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="附加信息（JSONB），避免使用 metadata 字段名与 SQLModel 冲突",
    )


class Document(SQLModelBase, table=True):
    """文档元数据"""

    __tablename__ = "notes_document"

    document_uid: str = Field(
        default_factory=lambda: uuid4().hex, index=True, unique=True
    )
    user_id: Optional[int] = Field(default=None, index=True)
    name: str
    source_type: str = Field(max_length=20)  # url / file / text / api
    source: Optional[str] = None
    version: int = Field(default=1)
    status: str = Field(default="active", max_length=20)  # active / disabled / deleted


class DocumentChunk(SQLModelBase, table=True):
    """文档分块"""

    __tablename__ = "notes_document_chunk"

    chunk_uid: str = Field(
        default_factory=lambda: uuid4().hex, index=True, unique=True
    )
    document_id: int = Field(index=True)
    chunk_index: int = Field(index=True)
    content: str
    embedding_id: Optional[str] = Field(
        default=None,
        index=True,
        description="在向量库中的主键，可选",
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="分块附加元信息（JSONB），如页码、标题等",
    )


class Memory(SQLModelBase, table=True):
    """长期记忆"""

    __tablename__ = "notes_memory"

    memory_uid: str = Field(
        default_factory=lambda: uuid4().hex, index=True, unique=True
    )
    user_id: Optional[int] = Field(default=None, index=True)
    conversation_id: Optional[int] = Field(default=None, index=True)
    memory_type: str = Field(max_length=50)  # preference / summary / fact / profile
    content: str
    score: float = Field(default=0.5)
