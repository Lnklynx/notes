import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.responses import error_response
from src.api.routes import chat, documents, health
from src.config import get_settings
from src.common.error_codes import get_http_status
from src.common.exceptions import BaseAppException
from src.db.session import init_db
from src.utils.logger import configure_sqlalchemy_logging, setup_logger

# 在应用模块级别获取一次 settings
settings = get_settings()

# 应用启动时配置日志系统
setup_logger("notes", level=logging.INFO if settings.debug else logging.WARNING)
if settings.debug:
    configure_sqlalchemy_logging(level=logging.INFO)
else:
    configure_sqlalchemy_logging(level=logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    logging.info("Application startup...")
    # 开发阶段：应用启动时自动创建缺失的数据表
    init_db()
    yield
    # 应用关闭时执行
    logging.info("Application shutdown...")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例"""

    app = FastAPI(
        title="AI Agent for Document Q&A",
        description="An intelligent agent for multi-turn document Q&A with ReAct pattern",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
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

    # 注册全局异常处理器 for 自定义业务异常
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        status_code = get_http_status(exc.error_code)
        response_content = error_response(
            code=exc.error_code.value,
            message=exc.message,
            data={"detail": exc.detail},
        )
        # 记录业务异常日志
        logging.warning(
            f"Business exception occurred: {exc.message}, detail: {exc.detail}"
        )
        return JSONResponse(
            status_code=status_code,
            content=response_content.model_dump(),
        )

    # 注册全局异常处理器 for 未捕获的系统异常
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        # 记录严重的服务器内部错误
        logging.error(
            f"An unhandled exception occurred: {exc}",
            exc_info=True,
        )
        response_content = error_response(
            code=90300,  # INTERNAL_SERVER_ERROR
            message="Internal Server Error",
            data={"detail": str(exc)} if settings.debug else None,
        )
        return JSONResponse(
            status_code=500,
            content=response_content.model_dump(),
        )

    return app


# 创建应用实例
app = create_app()

# 应用启动入口
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        # factory=False，因为我们已经创建了 app 实例
    )
