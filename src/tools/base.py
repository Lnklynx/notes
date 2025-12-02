from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """工具抽象基类"""

    name: str
    description: str
    parameters: dict  # 工具参数定义（JSON Schema 格式）

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass

    def to_schema(self) -> dict:
        """生成符合 OpenAI Function Calling 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


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

    def invoke_tool(self, name: str, **kwargs) -> Any:
        """获取并执行指定的工具"""
        tool = self.get(name)
        return tool.execute(**kwargs)

    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的标准定义 Schema（用于 LLM 工具调用）"""
        return [tool.to_schema() for tool in self.tools.values()]
