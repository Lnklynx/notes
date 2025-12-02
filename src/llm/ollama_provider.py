from typing import Iterator, Optional
import httpx
import json
from .base import BaseLLM, LLMResponse, ToolCall, LLMRequest


class OllamaLLM(BaseLLM):
    """Ollama 本地模型适配"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=300.0)

    def chat(self, request: LLMRequest) -> LLMResponse:
        """同步调用 Ollama API"""
        try:
            formatted_messages = self._format_messages(request.messages)
            
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": request.temperature,
                "stream": False
            }
            
            if request.tools:
                payload["tools"] = request.tools
                payload["tool_choice"] = "auto"
            
            payload.update(request.extra_params)
            
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            
            content = result["message"].get("content", "")
            tool_calls = []
            
            if "tool_calls" in result["message"]:
                for tc in result["message"]["tool_calls"]:
                    try:
                        arguments = tc.get("function", {}).get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        tool_calls.append(ToolCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=arguments,
                            id=tc.get("id")
                        ))
                    except Exception:
                        continue
            
            return LLMResponse(content=content, tool_calls=tool_calls)
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """流式调用 Ollama API（暂不支持工具调用）"""
        try:
            formatted_messages = self._format_messages(request.messages)
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": request.temperature,
                "stream": True
            }
            
            payload.update(request.extra_params)
            
            with self.client.stream("POST", "/api/chat", json=payload) as response:
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
