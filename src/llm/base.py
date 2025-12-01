from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLM(ABC):
    """LLM 抽象基类，定义统一接口"""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """同步调用，返回单个响应"""
        pass

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """流式调用，逐字符返回"""
        pass

