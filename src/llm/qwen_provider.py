from typing import Iterator
import json
from .base import BaseLLM, LLMResponse, ToolCall, LLMRequest


class QwenLLM(BaseLLM):
    """阿里云千问模型适配（通过 DashScope API）"""

    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        try:
            from dashscope import Generation
        except ImportError:
            raise ImportError("dashscope not installed. Install with: pip install dashscope")

        self.api_key = api_key
        self.model = model
        self.generation = Generation

        # 配置 API 密钥
        import dashscope
        dashscope.api_key = api_key

    def chat(self, request: LLMRequest) -> LLMResponse:
        """同步调用千问 API"""
        try:
            from dashscope import Generation

            kwargs = {
                "model": self.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": False
            }

            if request.tools:
                kwargs["tools"] = request.tools
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            if request.top_p:
                kwargs["top_p"] = request.top_p

            kwargs.update(request.extra_params)

            response = Generation.call(**kwargs)

            if response.status_code == 200:
                message = response.output.choices[0].message
                content = message.get("content", "")
                tool_calls = []

                if "tool_calls" in message:
                    for tc in message["tool_calls"]:
                        try:
                            function = tc.get("function", {})
                            arguments = function.get("arguments", {})
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                            tool_calls.append(ToolCall(
                                name=function.get("name", ""),
                                arguments=arguments,
                                id=tc.get("id")
                            ))
                        except Exception:
                            continue

                return LLMResponse(content=content, tool_calls=tool_calls)
            else:
                raise RuntimeError(f"Qwen API error: {response.message}")
        except Exception as e:
            raise RuntimeError(f"Qwen chat error: {str(e)}")

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """流式调用千问 API（暂不支持工具调用）"""
        try:
            from dashscope import Generation

            kwargs = {
                "model": self.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": True
            }

            if request.tools:
                kwargs["tools"] = request.tools
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            if request.top_p:
                kwargs["top_p"] = request.top_p

            kwargs.update(request.extra_params)

            responses = Generation.call(**kwargs)

            for response in responses:
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content
                else:
                    raise RuntimeError(f"Qwen streaming error: {response.message}")
        except Exception as e:
            raise RuntimeError(f"Qwen streaming error: {str(e)}")
