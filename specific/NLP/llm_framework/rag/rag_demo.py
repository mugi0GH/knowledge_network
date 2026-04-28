# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. 加载文档（确保同级目录有 test.txt）
loader = TextLoader("./opinions.txt")
docs = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 存储
embeddings = OllamaEmbeddings(model="qwen2.5:7b")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. 检索 + 生成
# llm = Ollama(model="qwen2.5:7b", temperature=0)
llm = OllamaLLM(model="qwen2.5:7b", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "这篇文章主要讲了什么？"
context_docs = retriever.invoke(query)  # 新版用 invoke
context = "\n\n".join([d.page_content for d in context_docs])

prompt = f"基于以下内容回答问题：\n{context}\n\n问题：{query}\n答案："
response = llm.invoke(prompt)
print(response)
