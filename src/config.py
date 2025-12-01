from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 服务配置
    app_env: str = "development"
    port: int = 8000
    debug: bool = True

    # LLM 配置
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    qwen_api_key: str = ""
    qwen_model: str = "qwen-turbo"

    # 向量数据库
    vector_store_type: str = "chromadb"
    vector_db_path: str = "./data/vectordb"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 向量化参数
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_top_k: int = 5

    # Agent 参数
    max_iterations: int = 10
    temperature: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()

