from pathlib import Path

from ..lib.web_fetcher import BrowserLikeFetcher


class DocumentLoader:
    """文档加载器：支持多种来源"""

    @staticmethod
    def load_from_url(url: str) -> str:
        """从 URL 加载网页内容"""
        try:
            # 使用浏览器模拟抓取工具获取 HTML
            html = BrowserLikeFetcher.fetch(url)

            # 简单的 HTML 文本提取
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return text
        except Exception as e:
            raise ValueError(f"Failed to load URL: {str(e)}")

    @staticmethod
    def load_from_file(file_path: str) -> str:
        """从文件加载内容"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        elif suffix == ".md":
            return path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        elif suffix == ".docx":
            return DocumentLoader._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """从 PDF 加载文本"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    @staticmethod
    def _load_docx(file_path: str) -> str:
        """从 Word 文档加载文本"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {str(e)}")

    @staticmethod
    def load(source: str, source_type: str = "text") -> str:
        """通用加载方法"""
        if source_type == "url":
            return DocumentLoader.load_from_url(source)
        elif source_type == "file":
            return DocumentLoader.load_from_file(source)
        elif source_type == "text":
            return source
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
