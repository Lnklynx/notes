# AI Agent 系统技术设计文档

## 1. 系统概述

本系统构建一个基于 ReAct 模式的智能 AI Agent，核心能力是**多模态文档理解与多轮深度问答**。系统通过向量化技术处理网页、文件及附件，利用 LangGraph 编排 Agent 行为，支持基于上下文的推理与追问。

## 2. 设计哲学

*   **分层解耦**：I/O 层（API）、控制层（Agent）、能力层（LLM/Embedding）、存储层（VectorDB）严格分离。
*   **模型无关性**：LLM 与 Embedding 模块设计为适配器模式，支持 OpenAI、Anthropic、Ollama 等多平台热切换。
*   **流程即代码**：使用 LangGraph 将业务逻辑（ReAct 循环、状态转换）显式化为图结构，而非隐式嵌套在代码中。
*   **配置驱动**：所有环境相关参数通过 `.env` 统一管理，遵循 12-Factor App 原则。

## 3. 系统架构

### 3.1 逻辑架构
```text
[Client] (Web/Mobile)
   ↓ HTTP/REST
[API Gateway] (FastAPI)
   ↓
[Agent Core] (LangGraph) <─→ [Memory] (Conversation State)
   ↓ ReAct Loop
[Tool Registry]
   ├─ [Document Loader] ─→ [Chunker] ─→ [Embedder] ─→ [Vector Store]
   └─ [LLM Service] (Multi-Provider Interface)
```

### 3.2 目录结构设计
```text
src/
├── api/            # 接口层 (FastAPI routes, schemas)
├── agent/          # 核心逻辑 (LangGraph state, nodes, graph)
├── llm/            # 模型适配 (BaseLLM, OpenAI, Ollama...)
├── embedding/      # 向量化服务 (Chunking, VectorDB client)
├── tools/          # 工具集 (Search, DocProcessors)
├── utils/          # 通用工具
└── config.py       # 配置加载 (.env处理)
```

## 4. 核心模块设计

### 4.1 Agent 模块 (Core)
*   **模式**：ReAct (Reasoning + Acting)。
*   **实现**：基于 `LangGraph` 构建状态机。
*   **状态定义 (`AgentState`)**：
    *   `messages`: 完整对话历史。
    *   `context`: 当前检索到的文档片段。
    *   `next_step`: 决策结果 (Action/Finish)。
*   **工作流**：`Input` -> `Think` -> `Decide` -> `Tool(Retrieve)` -> `Synthesize` -> `Output`。

### 4.2 向量化模块 (Embedding)
*   **职责**：文档 ETL (Extract, Transform, Load)。
*   **流程**：
    1.  **Loader**: 识别 URL/File 类型并加载文本。
    2.  **Chunker**: 使用 `RecursiveCharacterTextSplitter` 进行语义分块（含重叠）。
    3.  **Embedder**: 调用模型生成向量。
    4.  **Store**: 存入 ChromaDB/Qdrant，建立 `(vector, metadata)` 索引。

### 4.3 LLM 模块
*   **设计**：工厂模式 + 策略模式。
*   **接口**：`generate(prompt)`, `stream(prompt)`。
*   **扩展性**：新增模型只需继承基类并在工厂注册，无需修改业务逻辑。

### 4.4 Web API 模块
*   **框架**：FastAPI。
*   **主要端点**：
    *   `POST /upload`: 接收文件/链接，触发后台向量化任务。
    *   `POST /chat`: 处理用户 query，返回 Agent 响应（支持 SSE 流式）。
    *   `GET /history`: 获取会话历史。

## 5. 核心流程逻辑

### 5.1 文档处理流
1.  接收输入（File/URL）。
2.  计算文件 Hash（去重检查）。
3.  解析文本 -> 清洗 -> 分块。
4.  批量向量化 -> 写入向量库。
5.  返回 `doc_id`。

### 5.2 问答交互流 (RAG)
1.  用户输入 Query。
2.  **Agent 思考**：判断是否需要查阅文档。
3.  **Action**：调用 `VectorSearch` 工具，传入 Query 和 `doc_id`。
4.  **Observation**：获取 Top-K 相关文档块。
5.  **Synthesis**：结合历史上下文和文档块，生成回答。
6.  更新会话状态。

## 6. 配置管理

使用 `pydantic-settings` 读取 `.env` 文件，确保类型安全。

**.env 示例模板**
```ini
# 服务配置
APP_ENV=development
PORT=8000

# LLM 配置
LLM_PROVIDER=openai # openai | anthropic | ollama
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
# OLLAMA_BASE_URL=http://localhost:11434

# 向量数据库
VECTOR_STORE_TYPE=chroma # chroma | qdrant
VECTOR_DB_PATH=./data/vectordb

# Embedding 配置
EMBEDDING_MODEL=text-embedding-3-small
```

