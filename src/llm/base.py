from abc import ABC, abstractmethod
from typing import Iterator, Optional, Any
from dataclasses import dataclass, field


class ToolCall:
    """工具调用信息
    
    表示 LLM 请求调用的工具及其参数。
    
    Attributes:
        name: 工具名称，对应 Tool.name
        arguments: 工具调用参数字典，键为参数名，值为参数值
        id: 工具调用唯一标识，如果未提供则自动生成
    """

    def __init__(self, name: str, arguments: dict[str, Any], id: Optional[str] = None):
        """
        Args:
            name: 工具名称，对应 Tool.name
            arguments: 工具调用参数字典，键为参数名，值为参数值
            id: 工具调用唯一标识，如果未提供则自动生成
        """
        self.name = name
        self.arguments = arguments
        self.id = id or f"call_{self.name}"


@dataclass
class LLMResponse:
    """LLM 响应，包含文本和工具调用信息
    
    封装 LLM 返回的响应内容，包括文本回答和可能的工具调用请求。
    """

    content: str = field(
        metadata={"description": "LLM 生成的文本内容"}
    )
    tool_calls: list[ToolCall] = field(
        default_factory=list,
        metadata={"description": "工具调用列表，如果 LLM 请求调用工具则包含在此列表中"}
    )

    def has_tool_calls(self) -> bool:
        """检查响应中是否包含工具调用
        
        Returns:
            bool: 如果包含工具调用返回 True，否则返回 False
        """
        return len(self.tool_calls) > 0


@dataclass
class LLMRequest:
    """LLM 请求参数封装
    
    统一封装所有 LLM 调用所需的参数，包括消息、工具、温度等模型控制参数。
    """

    messages: list[dict] = field(
        metadata={"description": "消息列表，格式为 [{'role': 'user', 'content': '...'}]"}
    )
    tools: Optional[list[dict]] = field(
        default=None,
        metadata={"description": "工具定义列表（OpenAI Function Calling 格式），用于工具调用"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"description": "采样温度，控制输出的随机性。范围 0-2，值越大输出越随机，默认 0.7"}
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"description": "最大生成 token 数量，限制响应长度"}
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"description": "核采样参数，控制输出的多样性。范围 0-1，默认 None"}
    )
    frequency_penalty: Optional[float] = field(
        default=None,
        metadata={"description": "频率惩罚，减少重复内容的生成。范围 -2.0 到 2.0，正值减少重复"}
    )
    presence_penalty: Optional[float] = field(
        default=None,
        metadata={"description": "存在惩罚，鼓励模型谈论新话题。范围 -2.0 到 2.0，正值鼓励新话题"}
    )
    stop: Optional[list[str]] = field(
        default=None,
        metadata={"description": "停止序列列表，遇到这些字符串时停止生成"}
    )
    extra_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "额外参数，用于扩展不常用的参数"}
    )


class BaseLLM(ABC):
    """LLM 抽象基类，定义统一接口"""

    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        """同步调用，返回响应（支持工具调用）
        
        Args:
            request: LLMRequest 对象，包含消息、工具和模型控制参数
        
        Returns:
            LLMResponse: 包含文本内容和工具调用信息
        """
        pass

    @abstractmethod
    def stream(self, request: LLMRequest) -> Iterator[str]:
        """流式调用，逐字符返回
        
        Args:
            request: LLMRequest 对象，包含消息、工具和模型控制参数
        
        Yields:
            str: 逐字符或逐块的响应内容
        """
        pass
