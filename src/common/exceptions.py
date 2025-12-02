from typing import Optional

from src.common.error_codes import ErrorCode, get_error_message


class BaseAppException(Exception):
    """应用异常基类"""

    def __init__(
        self,
        error_code: ErrorCode,
        message: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        self.error_code = error_code
        self.message = message or get_error_message(error_code)
        self.detail = detail
        super().__init__(self.message)


class BusinessException(BaseAppException):
    """业务异常 - 客户端请求错误或业务逻辑错误"""

    pass


class SystemException(BaseAppException):
    """系统异常 - 服务器内部错误"""

    pass


# 文档相关异常
class DocumentLoadFailedException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DOCUMENT_LOAD_FAILED, detail=detail)


class DocumentParseFailedException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DOCUMENT_PARSE_FAILED, detail=detail)


class DocumentNotFoundException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DOCUMENT_NOT_FOUND, detail=detail)


class DocumentTypeUnsupportedException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DOCUMENT_TYPE_UNSUPPORTED, detail=detail)


class DocumentUploadFailedException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DOCUMENT_UPLOAD_FAILED, detail=detail)


class DocumentEmptyException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DOCUMENT_EMPTY, detail=detail)


# 对话相关异常
class ConversationNotFoundException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.CONVERSATION_NOT_FOUND, detail=detail)


class ConversationCreateFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.CONVERSATION_CREATE_FAILED, detail=detail)


class MessageSendFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.MESSAGE_SEND_FAILED, detail=detail)


class LLMCallFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.LLM_CALL_FAILED, detail=detail)


class LLMProviderUnsupportedException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.LLM_PROVIDER_UNSUPPORTED, detail=detail)


class VectorSearchFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.VECTOR_SEARCH_FAILED, detail=detail)


class AgentGraphBuildFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.AGENT_GRAPH_BUILD_FAILED, detail=detail)


# 向量库相关异常
class VectorStoreNotInitializedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.VECTOR_STORE_NOT_INITIALIZED, detail=detail)


class VectorStoreOperationFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.VECTOR_STORE_OPERATION_FAILED, detail=detail)


class EmbeddingFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.EMBEDDING_FAILED, detail=detail)


# 配置相关异常
class ConfigMissingException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.CONFIG_MISSING, detail=detail)


class ConfigInvalidException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.CONFIG_INVALID, detail=detail)


class APIKeyMissingException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.API_KEY_MISSING, detail=detail)


# 参数验证异常
class ParameterMissingException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.PARAMETER_MISSING, detail=detail)


class ParameterInvalidException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.PARAMETER_INVALID, detail=detail)


class ParameterFormatErrorException(BusinessException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.PARAMETER_FORMAT_ERROR, detail=detail)


# 数据库相关异常
class DatabaseConnectionFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DATABASE_CONNECTION_FAILED, detail=detail)


class DatabaseOperationFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DATABASE_OPERATION_FAILED, detail=detail)


class DatabaseQueryFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.DATABASE_QUERY_FAILED, detail=detail)


# 网络相关异常
class NetworkTimeoutException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.NETWORK_TIMEOUT, detail=detail)


class NetworkErrorException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.NETWORK_ERROR, detail=detail)


class APICallFailedException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.API_CALL_FAILED, detail=detail)


class ServiceUnavailableException(SystemException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(ErrorCode.SERVICE_UNAVAILABLE, detail=detail)
