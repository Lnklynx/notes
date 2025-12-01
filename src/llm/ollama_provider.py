from typing import Iterator
import httpx
import json
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama 本地模型适配"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=300.0)

    def chat(self, messages: list[dict]) -> str:
        """同步调用 Ollama API"""
        try:
            # 转换 LangChain 消息格式为 Ollama 格式
            formatted_messages = self._format_messages(messages)
            response = self.client.post(
                "/api/chat",
                json={"model": self.model, "messages": formatted_messages, "stream": False}
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """流式调用 Ollama API"""
        try:
            formatted_messages = self._format_messages(messages)
            with self.client.stream(
                    "POST",
                    "/api/chat",
                    json={"model": self.model, "messages": formatted_messages, "stream": True}
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama streaming error: {str(e)}")

    @staticmethod
    def _format_messages(messages: list[dict]) -> list[dict]:
        """转换消息格式，兼容 LangChain BaseMessage 对象"""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                # 已经是字典格式
                formatted.append(msg)
            elif hasattr(msg, "__class__"):
                # LangChain BaseMessage 对象
                role_map = {
                    "HumanMessage": "user",
                    "AIMessage": "assistant",
                    "SystemMessage": "system"
                }
                class_name = msg.__class__.__name__
                role = role_map.get(class_name, "user")
                formatted.append({"role": role, "content": msg.content})
        return formatted
