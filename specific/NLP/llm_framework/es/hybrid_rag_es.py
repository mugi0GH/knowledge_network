from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_elasticsearch import ElasticsearchStore

# ---------- 1. 加载文档并分块 ----------
loader = TextLoader("./opinions.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ---------- 2. 向量检索 (Chroma) ----------
embeddings = OllamaEmbeddings(model="qwen2.5:7b")
vectorstore = Chroma.from_documents(chunks, embeddings)

# ---------- 3. BM25 检索 (Elasticsearch) ----------
# 注意：ElasticsearchStore 需要指定 index_name，并且我们只使用其文本检索能力（不依赖向量）
es_store = ElasticsearchStore(
    index_name="hybrid_docs",
    embedding=embeddings,  # 虽然这里传了 embedding，但我们下面会直接用 es_client 做 BM25 查询
    es_url="http://localhost:9200",
    es_user="",
    es_password="",
    # 使用 SparseVectorRetrievalStrategy 可以强制使用 BM25，但为了简单，我们直接用原生 ES 查询
)
# 清空旧索引（开发时方便）
try:
    es_store.delete_index()
except:
    pass
# 写入文档（ElasticsearchStore 会自动创建索引）
es_store.add_documents(chunks)

# 获取原生 ES 客户端，用于自定义 BM25 查询
es_client = es_store.client


# ---------- 4. 定义混合检索函数 (RRF 融合) ----------
def hybrid_search(query: str, k: int = 3):
    # 向量检索
    vector_results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    # BM25 检索（通过 ES 的 match 查询）
    es_response = es_client.search(
        # index=es_store.index_name, body={"query": {"match": {"text": query}}, "size": k}
        index="hybrid_docs",
        body={"query": {"match": {"text": query}}, "size": k},
    )
    # 将 ES 结果转换为 Document 对象
    from langchain_core.documents import Document

    bm25_results = []
    for hit in es_response["hits"]["hits"]:
        doc = Document(
            page_content=hit["_source"]["text"],
            metadata=hit["_source"].get("metadata", {}),
        )
        bm25_results.append((doc, hit["_score"]))

    # RRF 融合（倒数排名融合）
    scores = {}
    for rank, (doc, _) in enumerate(vector_results, start=1):
        doc_id = doc.page_content  # 简单用内容做 ID，实际可用 metadata 里的唯一 id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)
    for rank, (doc, _) in enumerate(bm25_results, start=1):
        doc_id = doc.page_content
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)

    # 按 RRF 得分排序取 top k
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    # 从 vector_results 和 bm25_results 中找回 Document
    doc_map = {doc.page_content: doc for doc, _ in vector_results}
    doc_map.update({doc.page_content: doc for doc, _ in bm25_results})
    final_docs = [doc_map[doc_id] for doc_id in sorted_ids if doc_id in doc_map]
    return final_docs


# ---------- 5. LLM 生成 ----------
llm = OllamaLLM(model="qwen2.5:7b", temperature=0, seed=42)


# 在文件末尾，原有测试代码改为：
def get_retriever():
    """返回 hybrid_search 函数，供 Agent 中的 RAGTool 使用"""
    return hybrid_search


def get_document_sources() -> list[str]:
    """返回已索引的文档来源列表（去重）"""
    results = vectorstore.get(include=["metadatas"])
    sources = {m.get("source", "未知") for m in results["metadatas"] if m}
    return sorted(sources)


def get_llm():
    """返回 OllamaLLM 实例，供 Agent 复用"""
    return llm


if __name__ == "__main__":
    # 原有的测试代码
    query = "这篇文章主要讲了什么？"
    context_docs = hybrid_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"基于以下内容回答问题：\n{context}\n\n问题：{query}\n答案："
    response = llm.invoke(prompt)
    print(response)
