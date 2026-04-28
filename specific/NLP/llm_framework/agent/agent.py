import sys
import os

# 将项目根目录加入 Python 路径，以便导入 es 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from es.hybrid_rag_es import get_retriever, get_llm, get_document_sources

try:
    from tools import RAGTool, CalculatorTool, WebSearchTool, ChatTool  # direct run
except ImportError:
    from agent.tools import RAGTool, CalculatorTool, WebSearchTool, ChatTool  # module run

# ---------- 初始化 ----------
retriever = get_retriever()  # 返回 hybrid_search 函数
llm = get_llm()  # 返回 OllamaLLM 实例

rag_tool = RAGTool(retriever, llm, get_document_sources)
calc_tool = CalculatorTool(llm)
web_tool = WebSearchTool(llm)
chat_tool = ChatTool(llm)

tools = {
    "rag_query": rag_tool,
    "calculator": calc_tool,
    "web_search": web_tool,
    "chat": chat_tool,
}


# ---------- 意图识别（选择一种） ----------
# 方法 A：简单关键词匹配（推荐先使用）
def route_intent(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["计算", "加", "减", "乘", "除", "等于", "多少"]):
        return "calculator"
    elif any(kw in q for kw in ["天气", "新闻", "股价", "搜索"]):
        return "web_search"
    elif any(kw in q for kw in ["你好", "谢谢", "再见"]):
        return "chat"
    else:
        return "rag_query"


# 方法 B：LLM 意图分类（更智能，但会多一次 LLM 调用）
def classify_intent_with_llm(query: str) -> str:
    prompt = f"""判断以下用户问题属于哪一类，只输出类别名称。
类别：rag_query, calculator, web_search, chat
用户问题：{query}
类别："""
    response = llm.invoke(prompt).strip().lower()
    if response in ["rag_query", "calculator", "web_search", "chat"]:
        return response
    else:
        return "rag_query"


# ---------- Agent 执行器 ----------
def run_agent(query: str, use_llm_intent: bool = False):
    # 选择路由方式
    if use_llm_intent:
        intent = classify_intent_with_llm(query)
    else:
        intent = route_intent(query)

    print(f"[路由: {intent}]")
    tool = tools.get(intent)
    if tool is None:
        return "抱歉，无法处理该请求。"
    return tool.run(query)


# ---------- 主交互循环 ----------
if __name__ == "__main__":
    print("Agent 已启动（输入 exit 退出）")
    print("提示：默认使用关键词路由，若需 LLM 路由请修改代码中 use_llm_intent=True")
    while True:
        sys.stdout.write("\n问: ")
        sys.stdout.flush()
        raw = sys.stdin.buffer.readline()
        if not raw:
            break
        user_input = raw.decode("utf-8", errors="replace").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
        response = run_agent(user_input, use_llm_intent=True)
        print(f"答: {response}")
