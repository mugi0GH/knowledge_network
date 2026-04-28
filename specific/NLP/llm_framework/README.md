# LLM Framework

基于 LangChain + Ollama + Elasticsearch 的全本地化 RAG Agent 框架。支持混合检索（向量 + BM25 RRF 融合）、MCP 工具协议、Ollama function calling 自主路由，无任何外部 API 依赖。

## 项目结构

```
llm_framework/
├── agent/
│   ├── agent.py          # 传统 Agent：手写意图路由 + 工具调度（独立运行版）
│   ├── orchestrator.py   # MCP 版 Agent：Ollama function calling 自主路由
│   └── tools.py          # 工具实现：RAGTool / CalculatorTool / WebSearchTool / ChatTool
├── es/
│   └── hybrid_rag_es.py  # 混合检索核心：Chroma 向量 + ES BM25 + RRF 融合
├── rag/
│   └── rag_demo.py       # 纯向量 RAG 示例（Chroma only）
├── mcp_server.py         # MCP Server：将工具暴露为标准协议接口
├── opinions.txt          # 示例知识库文档
└── pixi.toml             # 依赖管理（pixi）
```

## 前置服务

**Ollama**（本地 LLM）：
```bash
ollama serve
ollama pull qwen2.5:7b
```

**Elasticsearch**（BM25 检索）：
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" elasticsearch:8.13.0
```

## 安装依赖

```bash
pixi install
```

主要依赖：`langchain` / `langchain-ollama` / `langchain-chroma` / `langchain-elasticsearch` / `ddgs` / `mcp`

## 快速开始

### 方式一：MCP Orchestrator（推荐）

Ollama function calling 自主决定调用哪个工具，无手写路由：

```bash
pixi run python agent/orchestrator.py
```

```
Agent 已启动，加载了 4 个工具
  - rag_query: 从本地知识库检索并回答问题
  - list_documents: 列出知识库中已索引的所有文档
  - calculate: 用自然语言描述数学问题，返回计算结果
  - web_search: 搜索互联网实时信息并返回 LLM 总结

问: 今天成都天气如何？
[调用工具: web_search]
答: 今天成都多云转晴，最高温度 37℃ ...

问: 10的四次方是多少
[调用工具: calculate]
答: 10000
```

### 方式二：传统 Agent（独立运行，无需 MCP）

```bash
pixi run python agent/agent.py
```

### 方式三：纯向量 RAG 示例

```bash
pixi run python rag/rag_demo.py
```

### 方式四：混合检索单独测试

```bash
pixi run python es/hybrid_rag_es.py
```

## MCP Server

[mcp_server.py](mcp_server.py) 将四个工具封装为标准 MCP 协议接口，可接入 Claude Desktop、Claude Code 等任意 MCP 客户端。

**注册到 Claude Desktop**（`~/.config/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "llm-framework": {
      "command": "/path/to/llm_framework/.pixi/envs/default/bin/python",
      "args": ["/path/to/llm_framework/mcp_server.py"]
    }
  }
}
```

重启 Claude Desktop 后，Claude 会自动调用本地知识库检索和计算工具。

## 混合检索原理

[es/hybrid_rag_es.py](es/hybrid_rag_es.py) 实现 **RRF（Reciprocal Rank Fusion）** 融合：

1. **向量检索**：Chroma + Ollama Embeddings，捕获语义相似性
2. **BM25 检索**：Elasticsearch `match` 查询，捕获关键词精确匹配
3. **RRF 融合**：`score = Σ 1/(60 + rank)`，合并两路排名取 top-k

## 扩展 Tool

在 [agent/tools.py](agent/tools.py) 中继承 `Tool` 基类：

```python
class MyTool(Tool):
    name = "my_tool"
    description = "..."  # MCP/LLM 路由依赖这段描述，写清楚触发条件

    def run(self, input: str) -> str:
        ...
```

MCP 版：在 [mcp_server.py](mcp_server.py) 中用 `@mcp.tool()` 注册即可，Ollama 会自动纳入路由候选。
