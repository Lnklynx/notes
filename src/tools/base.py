from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """工具抽象基类"""

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """获取工具"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered")
        return self.tools[name]

    def list_tools(self) -> list[dict]:
        """列出所有可用工具及说明"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

