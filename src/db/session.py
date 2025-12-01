from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import SQLModel, Session, create_engine

from ..config import get_settings


_engine = None


def get_engine():
    """惰性创建数据库引擎"""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        _engine = create_engine(url, echo=settings.debug, pool_pre_ping=True)
    return _engine


def init_db() -> None:
    """基于模型创建缺失的数据表（开发阶段使用）"""
    engine = get_engine()
    from . import models  # noqa: F401  确保模型被导入

    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """提供数据库会话上下文管理"""
    engine = get_engine()
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


