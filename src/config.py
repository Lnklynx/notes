from pydantic_settings import BaseSettings


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
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_dir: str | None = None

    # 向量化参数
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_top_k: int = 5

    # Agent 参数
    max_iterations: int = 10
    temperature: float = 0.7

    # 数据库配置
    database_url: str = "postgresql://postgres:admin123@localhost:5432/notes"

    # LangSmith (可选, 用于追踪和调试)
    langchain_tracing_v2: str = "false"
    langchain_api_key: str | None = None
    langchain_project: str | None = None
    langsmith_endpoint: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    return Settings()
