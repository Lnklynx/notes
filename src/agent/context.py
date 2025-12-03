from typing import Any
from pydantic import BaseModel, Field


class RetrievedContent(BaseModel):
    """
    标准化的检索内容片段。
    工具执行后，应将结果转换为此结构。
    """
    source_id: str = Field(..., description="来源标识 (如 document_uid, chunk_id)")
    content: str = Field(..., description="文本内容片段")
    score: float = Field(default=0.0, description="相关性分数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="其他元数据")


class AgentContext(BaseModel):
    """
    Agent 的工作上下文，作为其“短期记忆”和“工作台”。
    它随 Agent 的思考和行动而演变。
    """
    # 初始的用户意图，通常保持不变
    scope: 'ResourceScope' = Field(
        description="本次任务的初始资源范围"
    )
    
    # 由工具执行动态填充
    retrieved_content: list[RetrievedContent] = Field(
        default_factory=list,
        description="已检索到的相关内容片段，供 LLM 综合回答"
    )
    
    # 用于多步骤任务或复杂逻辑
    work_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent 的中间工作状态，如任务进度、临时变量等"
    )

# 避免循环导入的类型引用
from src.api.models import ResourceScope
AgentContext.model_rebuild()
