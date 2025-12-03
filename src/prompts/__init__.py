"""
Prompt 管理模块

集中管理项目中所有系统级别的 prompt 文本。
"""

from .prompts import (
    SystemPrompts,
    get_system_prompt,
)

__all__ = [
    "SystemPrompts",
    "get_system_prompt",
]

