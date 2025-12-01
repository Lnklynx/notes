#!/usr/bin/env python3
"""
API 测试脚本：演示如何通过 HTTP 调用系统
"""

import httpx
import json
import time
from typing import Optional

BASE_URL = "http://localhost:8000"
CONVERSATION_UID = "test_conversation_001"
DOCUMENT_UID = ""


def test_health():
    """测试健康检查"""
    print("\n" + "=" * 60)
    print("测试 1: 健康检查")
    print("=" * 60)

    try:
        response = httpx.get(f"{BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print(f"  提示: 确保服务已启动")
        print(f"  $ python main.py")
        return False


def test_upload_document():
    """测试文档上传"""
    global DOCUMENT_UID

    print("\n" + "=" * 60)
    print("测试 2: 文档上传")
    print("=" * 60)

    doc_content = """
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

    payload = {
        "content": doc_content,
        "source_type": "text",
        "name": "test_document"
    }

    try:
        response = httpx.post(
            f"{BASE_URL}/api/documents/upload",
            json=payload
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")

        if response.status_code == 200:
            DOCUMENT_UID = result.get("document_id")
            print(f"✓ 文档 UID: {DOCUMENT_UID}")
            return True
        return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def test_chat(message: str, stream: bool = False):
    """测试对话"""

    print(f"\n{'─' * 60}")
    print(f"问题: {message}")
    print(f"{'─' * 60}")

    if not DOCUMENT_UID:
        print("✗ 跳过: 文档 UID 为空，请先上传文档")
        return False

    payload = {
        "conversation_uid": CONVERSATION_UID,
        "document_uid": DOCUMENT_UID,
        "message": message,
        "stream": stream
    }

    try:
        response = httpx.post(
            f"{BASE_URL}/api/chat/send",
            json=payload,
            timeout=60.0
        )
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n回答: {result.get('answer')}")

            if result.get("documents"):
                print(f"\n参考文档 ({len(result['documents'])} 片段):")
                for doc in result["documents"][:2]:
                    print(f"  • {doc[:100]}...")
            return True
        else:
            print(f"错误: {response.json()}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def test_history():
    """测试获取对话历史"""
    print("\n" + "=" * 60)
    print("测试 4: 获取对话历史")
    print("=" * 60)

    try:
        response = httpx.get(
            f"{BASE_URL}/api/chat/history/{CONVERSATION_UID}"
        )
        print(f"状态码: {response.status_code}")
        result = response.json()

        messages = result.get("messages", [])
        print(f"对话轮数: {len(messages)}")

        for i, msg in enumerate(messages, 1):
            role = msg.get("role")
            content = msg.get("content", "")[:100]
            print(f"  {i}. [{role}]: {content}...")

        return True
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("AI Agent API 测试")
    print("本地模型: Ollama qwen3:8b")
    print("=" * 60)

    # 检查服务是否运行
    if not test_health():
        return

    # 上传文档
    if not test_upload_document():
        print("\n✗ 文档上传失败")
        return

    print("\n" + "=" * 60)
    print("测试 3: 多轮对话")
    print("=" * 60)

    # 多轮问答
    questions = [
        "Python 有哪些主要应用领域？",
        "什么是深度学习？",
        "机器学习和深度学习有什么区别？"
    ]

    for question in questions:
        if not test_chat(question):
            print("\n✗ 对话请求失败")
            break
        time.sleep(1)  # 避免请求过快

    # 获取历史
    test_history()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
