# TODO

## 高优先级

- [ ] **多轮对话记忆** — `orchestrator.py` 目前每次问答无状态，将 `messages` 历史在会话内持久传递，支持追问和上下文引用
- [ ] **MCP 文档索引工具** — `mcp_server.py` 新增 `index_file(path)` 工具，让模型能主动将新文件写入知识库，完成 RAG 闭环
- [ ] **FastAPI HTTP 接口** — 将 orchestrator 包装为 REST API（`POST /chat`），使 Agent 可作为独立服务部署而非仅 CLI

## 中优先级

- [ ] **LangGraph 重构** — 用 LangGraph 替换 `orchestrator.py` 中的手写 ReAct 循环，支持条件分支、并行工具调用、出错重试和有状态图
- [ ] **Reranker** — 混合检索后加 cross-encoder 重排（`sentence-transformers` 本地运行），提升召回精度，对齐生产级 RAG 标准配置
- [ ] **流式输出** — Ollama 支持 streaming，orchestrator 改为流式返回逐字输出，改善交互体验
- [ ] **RAG 评估** — 引入 RAGAS 指标（faithfulness / answer relevancy / context recall），构建测试集量化检索质量；做完后简历可写"有完整评估体系"

## 低优先级 / 探索性

- [ ] **多文档支持完善** — 支持 PDF / Word / Markdown 批量索引，目前仅支持 txt；增量索引避免重复写入
- [ ] **安全加固** — `CalculatorTool` 的 `eval` 替换为 `asteval` 或 `numexpr`，消除代码注入风险
- [ ] **日志与可观测性** — 统一结构化日志，记录工具调用链路、耗时、检索命中率

## LLM 工程能力补全

- [ ] **LoRA / QLoRA 微调** — 基于 Unsloth 对 Qwen2.5 做领域微调实验，理解训练数据格式、学习率调度、显存优化；做完后简历可写"有微调经验"
- [ ] **Prompt 工程系统化** — 实践 CoT、few-shot、structured output（JSON mode）、HyDE（假设文档嵌入）查询增强；补充 prompt injection 防御
- [ ] **上下文压缩** — 长对话时 token 超限处理：摘要压缩历史、滑动窗口、重要性过滤

## 性能优化

- [ ] **Embedding 批处理** — 当前文档索引逐条 embed，改为批量调用减少 Ollama 往返次数
- [ ] **检索缓存** — 对高频相同 query 结果缓存（TTL 短期），降低 ES + Chroma 重复检索开销
- [ ] **异步工具调用** — `orchestrator.py` 中多个工具调用目前串行，LLM 请求多工具时改为 `asyncio.gather` 并行执行
- [ ] **模型量化对比** — 测试 Qwen2.5 在 Q4_K_M / Q8_0 / FP16 下的推理速度与回答质量权衡，结合你的边缘设备部署经验输出对比报告
- [ ] **向量索引优化** — Chroma 默认 HNSW 参数（ef_construction、M）未调优，针对文档规模调整，减少检索延迟
