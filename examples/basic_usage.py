#!/usr/bin/env python3
"""
基础使用示例：直接集成到代码中使用
"""

from src.embedding.chunker import DocumentChunker
from src.embedding.embedder import TextEmbedder
from src.embedding.vector_store import VectorStore
from src.llm.factory import create_llm
from src.tools.base import ToolRegistry
from src.tools.search import VectorSearchTool
from src.agent.graph import create_agent_graph
from src.agent.state import AgentState
from langchain_core.messages import HumanMessage


def simple_example():
    """最简单的使用示例"""
    
    # 1. 初始化向量化服务
    embedder = TextEmbedder()
    chunker = DocumentChunker()
    vector_store = VectorStore("./data/vectordb")
    
    # 2. 处理文档
    doc = "Python 是一种编程语言。机器学习是 AI 的重要分支。"
    chunks = chunker.chunk(doc)
    embeddings = embedder.embed_batch(chunks)
    
    # 3. 存储文档
    vector_store.create_collection("docs")
    vector_store.add_documents(
        ids=["chunk_0", "chunk_1"],
        embeddings=embeddings,
        documents=chunks
    )
    
    # 4. 初始化 LLM（本地 Ollama）
    llm = create_llm(
        provider="ollama",
        base_url="http://localhost:11434",
        model="qwen3:8b"
    )
    
    # 5. 创建工具与 Agent
    tool_registry = ToolRegistry()
    search_tool = VectorSearchTool(vector_store, embedder)
    tool_registry.register(search_tool)
    
    graph = create_agent_graph(
        llm,
        tool_registry,
        tool_registry.list_tools()
    )
    
    # 6. 执行问答
    state: AgentState = {
        "messages": [HumanMessage(content="Python 是什么？")],
        "documents": [],
        "next_action": "search",
        "tool_results": {},
        "iteration_count": 0,
        "final_answer": None,
        "conversation_uid": "demo_conversation",
        "document_uid": "doc_1",
    }
    
    result = graph.invoke(state)
    print("回答:", result.get("final_answer"))


def custom_llm_example():
    """使用不同 LLM 的示例"""
    
    # OpenAI
    llm_openai = create_llm(
        provider="openai",
        api_key="sk-...",
        model="gpt-4o"
    )
    
    # 千问
    llm_qwen = create_llm(
        provider="qwen",
        api_key="sk-...",
        model="qwen-turbo"
    )
    
    # Ollama 本地
    llm_ollama = create_llm(
        provider="ollama",
        base_url="http://localhost:11434",
        model="llama2"
    )
    
    # 所有 LLM 使用统一接口
    messages = [{"role": "user", "content": "你好"}]
    
    # 同步调用
    response = llm_qwen.chat(messages)
    print(response)
    
    # 流式调用
    for chunk in llm_qwen.stream(messages):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    print("运行简单示例...")
    simple_example()

