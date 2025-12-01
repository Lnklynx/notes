from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """文本分块处理"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

    def chunk(self, text: str) -> list[str]:
        """分割文本为块"""
        return self.splitter.split_text(text)
