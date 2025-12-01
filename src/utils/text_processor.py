import re


class TextProcessor:
    """文本处理工具"""

    @staticmethod
    def clean_text(text: str) -> str:
        """清洁文本"""
        # 移除多余空白
        text = re.sub(r"\s+", " ", text).strip()
        # 移除特殊符号（保留中文、英文、数字、基本标点）
        text = re.sub(r"[^\w\s\u4e00-\u9fff\.,，。！？；：'\"']", "", text)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """归一化空白符"""
        lines = text.split("\n")
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(cleaned_lines)
