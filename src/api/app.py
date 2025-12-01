from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import documents, chat, health
from ..config import get_settings

settings = get_settings()


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(
        title="AI Agent for Document Q&A",
        description="An intelligent agent for multi-turn document Q&A with ReAct pattern",
        version="0.1.0",
        debug=settings.debug
    )
    
    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(health.router)
    app.include_router(documents.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    
    return app

