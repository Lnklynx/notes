from typing import Iterator
from openai import OpenAI
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI 模型适配"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        formatted_messages = self._format_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        formatted_messages = self._format_messages(messages)
        with self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=0.7,
            stream=True
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    @staticmethod
    def _format_messages(messages: list[dict]) -> list[dict]:
        """转换消息格式，兼容 LangChain BaseMessage 对象"""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(msg)
            elif hasattr(msg, "__class__"):
                role_map = {
                    "HumanMessage": "user",
                    "AIMessage": "assistant",
                    "SystemMessage": "system"
                }
                class_name = msg.__class__.__name__
                role = role_map.get(class_name, "user")
                formatted.append({"role": role, "content": msg.content})
        return formatted

