from typing import Iterator
from .base import BaseLLM


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

    def chat(self, messages: list[dict]) -> str:
        """同步调用千问 API"""
        try:
            from dashscope import Generation

            response = Generation.call(
                model=self.model,
                messages=messages,
                stream=False
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise RuntimeError(f"Qwen API error: {response.message}")
        except Exception as e:
            raise RuntimeError(f"Qwen chat error: {str(e)}")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """流式调用千问 API"""
        try:
            from dashscope import Generation

            responses = Generation.call(
                model=self.model,
                messages=messages,
                stream=True
            )

            for response in responses:
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content
                else:
                    raise RuntimeError(f"Qwen streaming error: {response.message}")
        except Exception as e:
            raise RuntimeError(f"Qwen streaming error: {str(e)}")
