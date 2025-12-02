from __future__ import annotations

from typing import Iterator
import logging

from sqlmodel import SQLModel, Session, create_engine

from ..config import get_settings
from ..utils.logger import configure_sqlalchemy_logging

_engine = None


def get_engine():
    """惰性创建数据库引擎"""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        
        # 配置 SQLAlchemy 日志：只在 DEBUG 模式下显示 SQL，否则只显示警告
        if settings.debug:
            configure_sqlalchemy_logging(level=logging.INFO)
        else:
            configure_sqlalchemy_logging(level=logging.WARNING)
        
        # echo=False，因为我们通过日志系统统一控制输出
        _engine = create_engine(url, echo=False, pool_pre_ping=True)
    return _engine


def init_db() -> None:
    """基于模型创建缺失的数据表（开发阶段使用）"""
    engine = get_engine()
    from . import models  # noqa: F401  确保模型被导入

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI 依赖使用的数据库会话生成器"""
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
