# Changelog

## [Unreleased]

### Added
- `mcp_server.py` — 基于 FastMCP 将四个工具（rag_query / list_documents / calculate / web_search）封装为 MCP Server，支持 Claude Desktop、Claude Code 等任意 MCP 客户端接入
- `agent/orchestrator.py` — MCP 版 Agent：以 MCP Client 连接 MCP Server，通过 Ollama function calling 让本地模型自主路由工具调用，替代手写意图分类
- `es/hybrid_rag_es.py` 新增 `get_document_sources()` — 查询 Chroma 返回已索引文档来源列表（去重）
- `pixi.toml` 新增 `mcp` 依赖

### Changed
- `CalculatorTool` 改用 LLM 将自然语言数学问题转为 Python 表达式再 eval，支持"10的4次方"等中文描述；新增 `llm` 构造参数
- `WebSearchTool` 改用 LLM 对搜索结果去噪后总结回答；新增 `llm` 构造参数；搜索条数从 3 增至 5
- `RAGTool` 新增 `get_sources` 构造参数，检索结果带来源元数据（`[来源: xxx]`）；支持"哪些文档"类问题直接返回文档列表
- Agent 默认路由从关键词匹配切换为 LLM 意图分类（`use_llm_intent=True`）
- `pixi.toml` Web 搜索依赖从 `duckduckgo-search` 更新为 `ddgs`

### Fixed
- `agent.py` 相对导入（`from .tools import`）在直接运行时报 `ImportError`，改为 try/except 兼容绝对导入与相对导入两种场景
- `agent.py` 绝对导入时 `agent/` 目录下的 `agent.py` 自身被识别为 `agent` 模块，导致 `ModuleNotFoundError: 'agent' is not a package`，已修复
- `agent.py` / `orchestrator.py` 终端中文输入 `UnicodeDecodeError`，改用 `sys.stdin.buffer.readline()` + 手动 decode 替代 `input()` 和 `TextIOWrapper`
- `tools.py` `RAGTool.run()` 调用 `self.retriever.invoke()` 报 `AttributeError`（`get_retriever()` 返回普通函数），改为直接调用 `self.retriever(query)`
- `tools.py` `WebSearchTool` 包名从 `duckduckgo_search` 改为 `ddgs`，加 try/except 向后兼容

---

## [0.1.0] — 2025-04-21 (Initial Commit)

### Added
- `agent/agent.py` — Agent 主入口，支持意图路由（关键词 + LLM 分类）+ 多工具调度
- `agent/tools.py` — 四个 Tool 实现：RAGTool、CalculatorTool、WebSearchTool、ChatTool
- `es/hybrid_rag_es.py` — 混合检索核心：Chroma 向量检索 + Elasticsearch BM25 + RRF 融合
- `rag/rag_demo.py` — 纯向量 RAG 示例（Chroma only）
- `README.md` — 项目文档
