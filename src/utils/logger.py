import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    COLORS = {
        'DEBUG': '\033[36m',  # 青色
        'INFO': '\033[32m',  # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',  # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
        name: str = "notes",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        enable_color: bool = True,
) -> logging.Logger:
    """配置日志系统
    
    Args:
        name: 日志名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        enable_color: 是否启用颜色（仅控制台）
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 统一使用管道符分隔的格式
    log_format = '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s'

    if enable_color and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# 创建默认日志实例
default_logger = setup_logger()


def configure_sqlalchemy_logging(level: int = logging.WARNING):
    """配置 SQLAlchemy 日志级别，避免输出过多 SQL 语句
    
    Args:
        level: SQLAlchemy 引擎日志级别，默认 WARNING（只显示警告和错误）
    """
    # 控制 SQLAlchemy 引擎日志
    logging.getLogger('sqlalchemy.engine').setLevel(level)
    logging.getLogger('sqlalchemy.pool').setLevel(level)
    logging.getLogger('sqlalchemy.dialects').setLevel(level)

    # 为 SQLAlchemy 日志也应用统一格式
    sqlalchemy_logger = logging.getLogger('sqlalchemy')
    sqlalchemy_logger.setLevel(level)

    # 如果已有处理器，更新格式；否则添加新处理器
    if not sqlalchemy_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        sqlalchemy_logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志实例
    
    Args:
        name: 日志名称，如果为 None 则使用调用模块的名称
    
    Returns:
        Logger 实例
    """
    if name:
        return logging.getLogger(name)

    # 获取调用者的模块名
    import inspect
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'notes')
    return logging.getLogger(module_name)
