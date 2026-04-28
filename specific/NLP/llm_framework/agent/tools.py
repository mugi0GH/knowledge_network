from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, input: str) -> str:
        pass


class RAGTool(Tool):
    name = "rag_query"
    description = (
        "从知识库中检索信息并回答用户问题。适用于询问文档内容、技术细节、内部知识等。"
    )

    def __init__(self, retriever, llm, get_sources=None):
        self.retriever = retriever
        self.llm = llm
        self.get_sources = get_sources

    def run(self, query: str) -> str:
        # 列出可用文档
        if self.get_sources and any(
            kw in query for kw in ["哪些文档", "有什么文档", "文档列表", "可以访问"]
        ):
            sources = self.get_sources()
            if not sources:
                return "当前知识库中没有已索引的文档。"
            return "当前已索引的文档：\n" + "\n".join(f"- {s}" for s in sources)

        docs = self.retriever(query)
        if not docs:
            return "知识库中未找到相关内容。"
        context = "\n\n".join(
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        )
        prompt = f"基于以下知识库内容回答问题（如无相关内容请说明）：\n{context}\n\n问题：{query}\n答案："
        return self.llm.invoke(prompt)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "执行数学计算，支持加减乘除、幂运算等。输入应为数学表达式，如 '123+456'。"
    )

    def __init__(self, llm):
        self.llm = llm

    def run(self, expr: str) -> str:
        try:
            prompt = (
                f"将下面的数学问题转换为 Python 表达式，只输出表达式本身，不要任何解释或文字：\n{expr}"
            )
            py_expr = self.llm.invoke(prompt).strip()
            result = eval(py_expr)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"


class WebSearchTool(Tool):
    name = "web_search"
    description = "搜索互联网实时信息，如新闻、天气、股价等。输入为搜索关键词。"

    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str) -> str:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "未找到相关信息。"
        snippets = "\n".join(
            f"[{r['title']}] {r['body']}" for r in results
        )
        prompt = (
            f'以下是搜索"{query}"得到的原始片段，其中可能有无关内容，'
            f'请提取有用信息，用简洁的中文回答用户的问题，不要重复原文：\n\n{snippets}'
        )
        return self.llm.invoke(prompt)


class ChatTool(Tool):
    name = "chat"
    description = "普通对话，用于问候、闲聊等不需要检索或计算的情况。"

    def __init__(self, llm):
        self.llm = llm

    def run(self, input: str) -> str:
        return self.llm.invoke(input)
