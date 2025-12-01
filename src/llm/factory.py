from .base import BaseLLM
from .openai_provider import OpenAILLM
from .ollama_provider import OllamaLLM
from .qwen_provider import QwenLLM


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """LLM 工厂：根据 provider 创建对应实例"""

    provider = provider.lower().strip()
    
    if provider == "openai":
        return OpenAILLM(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "gpt-4o")
        )
    elif provider == "ollama":
        return OllamaLLM(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama2")
        )
    elif provider == "qwen":
        return QwenLLM(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "qwen-turbo")
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. "
                        f"Supported: openai, ollama, qwen")

