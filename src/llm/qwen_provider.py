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

            formatted_messages = self._format_messages(request.messages)

            kwargs = {
                "model": self.model,
                "messages": formatted_messages,
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

            formatted_messages = self._format_messages(request.messages)

            kwargs = {
                "model": self.model,
                "messages": formatted_messages,
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

    @staticmethod
    def _format_messages(messages: list) -> list[dict]:
        """转换消息格式，兼容 LangChain BaseMessage 对象
        
        将 LangChain 消息对象转换为 dashscope API 期望的字典格式。
        支持 HumanMessage, AIMessage, SystemMessage, ToolMessage。
        
        注意：dashscope API 要求 ToolMessage 必须紧跟在包含 tool_calls 的 AIMessage 之后。
        """
        formatted = []
        prev_assistant_has_tool_calls = False
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                # 已经是字典格式
                role = msg.get("role")
                
                if role == "tool":
                    # ToolMessage 必须紧跟在包含 tool_calls 的 AIMessage 之后
                    if not prev_assistant_has_tool_calls:
                        # 跳过这个 ToolMessage，因为前面没有包含 tool_calls 的 assistant 消息
                        continue
                    formatted.append(msg)
                    prev_assistant_has_tool_calls = False
                else:
                    formatted.append(msg)
                    # 检查是否是包含 tool_calls 的 assistant 消息
                    if role == "assistant" and "tool_calls" in msg:
                        prev_assistant_has_tool_calls = True
                    else:
                        prev_assistant_has_tool_calls = False
            elif hasattr(msg, "__class__"):
                # LangChain BaseMessage 对象
                class_name = msg.__class__.__name__

                if class_name == "HumanMessage":
                    formatted.append({
                        "role": "user",
                        "content": msg.content
                    })
                    prev_assistant_has_tool_calls = False
                elif class_name == "AIMessage":
                    msg_dict = {
                        "role": "assistant",
                        "content": msg.content or ""
                    }
                    # 处理工具调用
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls = []
                        for tc in msg.tool_calls:
                            # 兼容字典格式和对象格式的 tool_calls
                            if isinstance(tc, dict):
                                tool_call = {
                                    "id": tc.get("id", f"call_{tc.get('name', 'unknown')}"),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": json.dumps(tc.get("args", {})) if isinstance(tc.get("args"), dict) else str(tc.get("args", ""))
                                    }
                                }
                            else:
                                tool_call = {
                                    "id": getattr(tc, "id", f"call_{getattr(tc, 'name', 'unknown')}"),
                                    "type": "function",
                                    "function": {
                                        "name": getattr(tc, "name", ""),
                                        "arguments": json.dumps(getattr(tc, "arguments", getattr(tc, "args", {}))) if isinstance(
                                            getattr(tc, "arguments", getattr(tc, "args", {})), dict) else str(getattr(tc, "arguments", getattr(tc, "args", "")))
                                    }
                                }
                            tool_calls.append(tool_call)
                        msg_dict["tool_calls"] = tool_calls
                        prev_assistant_has_tool_calls = True
                    else:
                        prev_assistant_has_tool_calls = False
                    formatted.append(msg_dict)
                elif class_name == "SystemMessage":
                    formatted.append({
                        "role": "system",
                        "content": msg.content
                    })
                    # SystemMessage 不会重置 prev_assistant_has_tool_calls，因为它可以在任何位置
                elif class_name == "ToolMessage":
                    # ToolMessage 必须紧跟在包含 tool_calls 的 AIMessage 之后
                    # 如果前一条消息不是包含 tool_calls 的 assistant，则跳过此 ToolMessage
                    if not prev_assistant_has_tool_calls:
                        # 跳过这个 ToolMessage，因为前面没有包含 tool_calls 的 assistant 消息
                        continue
                    
                    tool_call_id = getattr(msg, "tool_call_id", "")
                    formatted.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": tool_call_id
                    })
                    # ToolMessage 后，重置标志（下一个 ToolMessage 需要新的 assistant 消息）
                    prev_assistant_has_tool_calls = False
                else:
                    # 未知类型，默认作为 user 消息
                    formatted.append({
                        "role": "user",
                        "content": str(msg.content) if hasattr(msg, "content") else str(msg)
                    })
                    prev_assistant_has_tool_calls = False
        return formatted
