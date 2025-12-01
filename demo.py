#!/usr/bin/env python3
"""
演示脚本：使用本地 Ollama qwen3:8b 模型进行文档问答
"""

import sys
from src.config import Settings
from src.embedding.chunker import DocumentChunker
from src.embedding.embedder import TextEmbedder
from src.embedding.vector_store import VectorStore
from src.llm.factory import create_llm
from src.tools.base import ToolRegistry
from src.tools.search import VectorSearchTool
from src.agent.graph import create_agent_graph
from src.agent.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


def demo_upload_document():
    """演示文档上传和向量化"""
    print("\n" + "="*60)
    print("步骤 1: 文档上传与向量化")
    print("="*60)
    
    # 创建示例文档
    sample_doc = """
    Python 是一种高级编程语言，具有简洁易学的语法。
    Python 广泛应用于数据科学、Web 开发、自动化脚本等领域。
    Python 社区活跃，拥有丰富的第三方库生态。
    
    机器学习是人工智能的一个重要分支。
    通过机器学习，计算机可以从数据中自动学习规律。
    常见的机器学习算法包括决策树、随机森林、神经网络等。
    
    深度学习是机器学习的一种方法，基于神经网络。
    深度学习在图像识别、自然语言处理等领域取得了重大成就。
    GPU 的发展加速了深度学习的应用。
    """
    
    # 初始化服务
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    embedder = TextEmbedder()
    vector_store = VectorStore("./data/vectordb_demo")
    
    # 分块
    print("\n➤ 分块文本...")
    chunks = chunker.chunk(sample_doc)
    print(f"   分块数量: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"\n   块 {i}: {chunk[:80]}...")
    
    # 向量化
    print("\n➤ 向量化处理...")
    embeddings = embedder.embed_batch(chunks)
    print(f"   向量维度: {len(embeddings[0])}")
    
    # 存储
    print("\n➤ 存储到向量库...")
    vector_store.create_collection("demo_docs")
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    vector_store.add_documents(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"doc_name": "sample.txt"} for _ in chunks]
    )
    vector_store.persist()
    print("   ✓ 存储完成")
    
    return vector_store, embedder


def demo_qa_with_ollama(vector_store, embedder):
    """演示使用 Ollama qwen3:8b 进行问答"""
    print("\n" + "="*60)
    print("步骤 2: 多轮问答交互")
    print("="*60)
    
    # 创建 LLM（使用本地 Ollama）
    print("\n➤ 初始化 LLM (Ollama qwen3:8b)...")
    try:
        llm = create_llm(
            provider="ollama",
            base_url="http://localhost:11434",
            model="qwen3:8b"
        )
        print("   ✓ LLM 连接成功")
    except Exception as e:
        print(f"   ✗ 连接失败: {e}")
        print("\n   提示: 请确保 Ollama 服务已启动:")
        print("   $ ollama serve")
        print("   $ ollama pull qwen3:8b  # 如果还未下载模型")
        return
    
    # 创建工具
    tool_registry = ToolRegistry()
    search_tool = VectorSearchTool(vector_store, embedder)
    tool_registry.register(search_tool)
    tools_info = tool_registry.list_tools()
    
    # 创建 Agent 图
    graph = create_agent_graph(
        llm,
        tool_registry,
        tools_info,
        max_iterations=3
    )
    
    # 交互轮次
    questions = [
        "Python 有哪些主要应用领域？",
        "什么是深度学习？",
        "机器学习和深度学习有什么区别？"
    ]
    
    messages = []
    
    for idx, question in enumerate(questions, 1):
        print(f"\n{'─'*60}")
        print(f"问题 {idx}: {question}")
        print(f"{'─'*60}")
        
        # 添加用户消息
        messages.append(HumanMessage(content=question))
        
        # 执行 Agent
        state: AgentState = {
            "messages": messages,
            "documents": [],
            "next_action": "search",
            "tool_results": {},
            "iteration_count": 0,
            "final_answer": None,
            "conversation_uid": "demo_session",
            "document_uid": "sample_doc",
        }
        
        print("\n➤ Agent 处理中...")
        try:
            result = graph.invoke(state)
            answer = result.get("final_answer", "无法生成回答")
            
            # 添加 AI 回复
            messages.append(AIMessage(content=answer))
            
            print(f"\n回答: {answer}")
            
            # 显示参考文档
            if result.get("documents"):
                print("\n参考文档片段:")
                for doc in result["documents"][:2]:
                    print(f"  • {doc[:100]}...")
        except Exception as e:
            print(f"\n✗ 处理出错: {e}")
            import traceback
            traceback.print_exc()


def demo_stream_response():
    """演示流式响应"""
    print("\n" + "="*60)
    print("步骤 3: 流式响应演示")
    print("="*60)
    
    print("\n➤ 初始化流式 LLM...")
    try:
        llm = create_llm(
            provider="ollama",
            base_url="http://localhost:11434",
            model="qwen3:8b"
        )
        print("   ✓ LLM 连接成功\n")
    except Exception as e:
        print(f"   ✗ 连接失败: {e}")
        return
    
    question = "请简要介绍什么是人工智能？"
    print(f"问题: {question}\n")
    print("回答 (流式):")
    print("─" * 40)
    
    messages = [{"role": "user", "content": question}]
    
    try:
        for chunk in llm.stream(messages):
            print(chunk, end="", flush=True)
        print("\n" + "─" * 40)
    except Exception as e:
        print(f"\n✗ 流式处理出错: {e}")


def main():
    print("\n" + "="*60)
    print("AI Agent 演示程序")
    print("本地模型: Ollama qwen3:8b")
    print("="*60)
    
    # 步骤 1: 文档上传
    vector_store, embedder = demo_upload_document()
    
    # 步骤 2: 多轮问答
    if vector_store and embedder:
        demo_qa_with_ollama(vector_store, embedder)
    
    # 步骤 3: 流式响应
    demo_stream_response()
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

