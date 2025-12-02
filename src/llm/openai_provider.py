from typing import Iterator, Optional
from openai import OpenAI
import json
from .base import BaseLLM, LLMResponse, ToolCall, LLMRequest


class OpenAILLM(BaseLLM):
    """OpenAI 模型适配"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, request: LLMRequest) -> LLMResponse:
        formatted_messages = self._format_messages(request.messages)
        
        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": request.temperature
        }
        
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.top_p:
            kwargs["top_p"] = request.top_p
        if request.frequency_penalty:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.stop:
            kwargs["stop"] = request.stop
        
        kwargs.update(request.extra_params)
        
        response = self.client.chat.completions.create(**kwargs)
        
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []
        
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=arguments,
                        id=tc.id
                    ))
                except Exception:
                    continue
        
        return LLMResponse(content=content, tool_calls=tool_calls)

    def stream(self, request: LLMRequest) -> Iterator[str]:
        formatted_messages = self._format_messages(request.messages)
        
        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": request.temperature,
            "stream": True
        }
        
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.top_p:
            kwargs["top_p"] = request.top_p
        if request.frequency_penalty:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.stop:
            kwargs["stop"] = request.stop
        
        kwargs.update(request.extra_params)
        
        with self.client.chat.completions.create(**kwargs) as stream:
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

