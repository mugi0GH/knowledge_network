import sys
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])

sys.path.insert(0, __import__("os").path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from es.hybrid_rag_es import get_retriever, get_llm, get_document_sources
from agent.tools import RAGTool, CalculatorTool, WebSearchTool

retriever = get_retriever()
llm = get_llm()
rag_tool = RAGTool(retriever, llm, get_document_sources)
calc_tool = CalculatorTool(llm)
web_tool = WebSearchTool(llm)

mcp = FastMCP("llm-framework")


@mcp.tool()
def rag_query(query: str) -> str:
    """从本地知识库检索并回答问题（混合向量+BM25检索）"""
    return rag_tool.run(query)


@mcp.tool()
def list_documents() -> str:
    """列出知识库中已索引的所有文档"""
    sources = get_document_sources()
    if not sources:
        return "知识库中暂无已索引文档。"
    return "已索引文档：\n" + "\n".join(f"- {s}" for s in sources)


@mcp.tool()
def calculate(expression: str) -> str:
    """用自然语言描述数学问题，返回计算结果"""
    return calc_tool.run(expression)


@mcp.tool()
def web_search(query: str) -> str:
    """搜索互联网实时信息并返回 LLM 总结"""
    return web_tool.run(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
