from enum import IntEnum
from typing import Dict


class ErrorCode(IntEnum):
    """错误码枚举 - 业务异常 (10000-19999)"""

    # 文档相关 (10001-10099)
    DOCUMENT_LOAD_FAILED = 10001
    DOCUMENT_PARSE_FAILED = 10002
    DOCUMENT_NOT_FOUND = 10003
    DOCUMENT_TYPE_UNSUPPORTED = 10004
    DOCUMENT_UPLOAD_FAILED = 10005
    DOCUMENT_EMPTY = 10006

    # 对话相关 (10100-10199)
    CONVERSATION_NOT_FOUND = 10100
    CONVERSATION_CREATE_FAILED = 10101
    MESSAGE_SEND_FAILED = 10102
    LLM_CALL_FAILED = 10103
    LLM_PROVIDER_UNSUPPORTED = 10104
    VECTOR_SEARCH_FAILED = 10105
    AGENT_GRAPH_BUILD_FAILED = 10106

    # 向量库相关 (10200-10299)
    VECTOR_STORE_NOT_INITIALIZED = 10200
    VECTOR_STORE_OPERATION_FAILED = 10201
    EMBEDDING_FAILED = 10202

    # 配置相关 (10300-10399)
    CONFIG_MISSING = 10300
    CONFIG_INVALID = 10301
    API_KEY_MISSING = 10302

    # 系统异常 (90000-99999)
    # 参数验证 (90001-90099)
    PARAMETER_MISSING = 90001
    PARAMETER_INVALID = 90002
    PARAMETER_FORMAT_ERROR = 90003

    # 数据库错误 (90100-90199)
    DATABASE_CONNECTION_FAILED = 90100
    DATABASE_OPERATION_FAILED = 90101
    DATABASE_QUERY_FAILED = 90102

    # 网络错误 (90200-90299)
    NETWORK_TIMEOUT = 90200
    NETWORK_ERROR = 90201
    API_CALL_FAILED = 90202
    SERVICE_UNAVAILABLE = 90203

    # 内部错误 (90300-90399)
    INTERNAL_SERVER_ERROR = 90300
    UNKNOWN_ERROR = 90399


ERROR_MESSAGES: Dict[ErrorCode, str] = {
    # 文档相关
    ErrorCode.DOCUMENT_LOAD_FAILED: "文档加载失败",
    ErrorCode.DOCUMENT_PARSE_FAILED: "文档解析失败",
    ErrorCode.DOCUMENT_NOT_FOUND: "文档不存在",
    ErrorCode.DOCUMENT_TYPE_UNSUPPORTED: "不支持的文件类型",
    ErrorCode.DOCUMENT_UPLOAD_FAILED: "文档上传失败",
    ErrorCode.DOCUMENT_EMPTY: "文档内容为空",
    # 对话相关
    ErrorCode.CONVERSATION_NOT_FOUND: "会话不存在",
    ErrorCode.CONVERSATION_CREATE_FAILED: "会话创建失败",
    ErrorCode.MESSAGE_SEND_FAILED: "消息发送失败",
    ErrorCode.LLM_CALL_FAILED: "大语言模型调用失败",
    ErrorCode.LLM_PROVIDER_UNSUPPORTED: "不支持的大语言模型提供者",
    ErrorCode.VECTOR_SEARCH_FAILED: "向量检索失败",
    ErrorCode.AGENT_GRAPH_BUILD_FAILED: "Agent 图构建失败",
    # 向量库相关
    ErrorCode.VECTOR_STORE_NOT_INITIALIZED: "向量库未初始化",
    ErrorCode.VECTOR_STORE_OPERATION_FAILED: "向量库操作失败",
    ErrorCode.EMBEDDING_FAILED: "向量化失败",
    # 配置相关
    ErrorCode.CONFIG_MISSING: "配置项缺失",
    ErrorCode.CONFIG_INVALID: "配置项无效",
    ErrorCode.API_KEY_MISSING: "API密钥缺失",
    # 参数验证
    ErrorCode.PARAMETER_MISSING: "请求参数缺失",
    ErrorCode.PARAMETER_INVALID: "请求参数无效",
    ErrorCode.PARAMETER_FORMAT_ERROR: "请求参数格式错误",
    # 数据库错误
    ErrorCode.DATABASE_CONNECTION_FAILED: "数据库连接失败",
    ErrorCode.DATABASE_OPERATION_FAILED: "数据库操作失败",
    ErrorCode.DATABASE_QUERY_FAILED: "数据库查询失败",
    # 网络错误
    ErrorCode.NETWORK_TIMEOUT: "网络请求超时",
    ErrorCode.NETWORK_ERROR: "网络错误",
    ErrorCode.API_CALL_FAILED: "API调用失败",
    ErrorCode.SERVICE_UNAVAILABLE: "服务不可用",
    # 内部错误
    ErrorCode.INTERNAL_SERVER_ERROR: "内部服务器错误",
    ErrorCode.UNKNOWN_ERROR: "未知错误",
}

ERROR_HTTP_STATUS: Dict[ErrorCode, int] = {
    # 业务异常 - 通常返回 400
    ErrorCode.DOCUMENT_LOAD_FAILED: 400,
    ErrorCode.DOCUMENT_PARSE_FAILED: 400,
    ErrorCode.DOCUMENT_NOT_FOUND: 404,
    ErrorCode.DOCUMENT_TYPE_UNSUPPORTED: 400,
    ErrorCode.DOCUMENT_UPLOAD_FAILED: 400,
    ErrorCode.DOCUMENT_EMPTY: 400,
    ErrorCode.CONVERSATION_NOT_FOUND: 404,
    ErrorCode.CONVERSATION_CREATE_FAILED: 500,
    ErrorCode.MESSAGE_SEND_FAILED: 500,
    ErrorCode.LLM_CALL_FAILED: 500,
    ErrorCode.LLM_PROVIDER_UNSUPPORTED: 400,
    ErrorCode.VECTOR_SEARCH_FAILED: 500,
    ErrorCode.AGENT_GRAPH_BUILD_FAILED: 500,
    ErrorCode.VECTOR_STORE_NOT_INITIALIZED: 500,
    ErrorCode.VECTOR_STORE_OPERATION_FAILED: 500,
    ErrorCode.EMBEDDING_FAILED: 500,
    ErrorCode.CONFIG_MISSING: 500,
    ErrorCode.CONFIG_INVALID: 500,
    ErrorCode.API_KEY_MISSING: 500,
    # 参数验证 - 返回 400
    ErrorCode.PARAMETER_MISSING: 400,
    ErrorCode.PARAMETER_INVALID: 400,
    ErrorCode.PARAMETER_FORMAT_ERROR: 400,
    # 数据库错误 - 返回 500
    ErrorCode.DATABASE_CONNECTION_FAILED: 500,
    ErrorCode.DATABASE_OPERATION_FAILED: 500,
    ErrorCode.DATABASE_QUERY_FAILED: 500,
    # 网络错误 - 返回 502 或 503
    ErrorCode.NETWORK_TIMEOUT: 504,
    ErrorCode.NETWORK_ERROR: 502,
    ErrorCode.API_CALL_FAILED: 502,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    # 内部错误 - 返回 500
    ErrorCode.INTERNAL_SERVER_ERROR: 500,
    ErrorCode.UNKNOWN_ERROR: 500,
}


def get_error_message(error_code: ErrorCode, default: str = None) -> str:
    """获取错误信息"""
    return ERROR_MESSAGES.get(error_code, default or "未知错误")


def get_http_status(error_code: ErrorCode) -> int:
    """获取错误码对应的HTTP状态码"""
    return ERROR_HTTP_STATUS.get(error_code, 500)
