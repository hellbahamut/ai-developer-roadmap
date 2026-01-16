# 5.4 RAG（检索增强生成）

## 1. RAG基础概念

**为什么需要RAG？**
- LLM知识有截止日期
- LLM不知道私有数据
- LLM会产生幻觉

**RAG原理**：
1. 用户问题
2. 检索相关文档（向量搜索）
3. 将文档作为上下文
4. LLM生成答案

## 2. 文档加载与处理

```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载PDF
pdf_loader = PyPDFLoader("document.pdf")
# pdf_pages = pdf_loader.load()

# 2. 加载文本文件
text_loader = TextLoader("article.txt", encoding='utf-8')
# text_docs = text_loader.load()

# 3. 加载整个目录
directory_loader = DirectoryLoader(
    './documents',
    glob="**/*.txt",
    loader_cls=TextLoader
)
# docs = directory_loader.load()

# 4. 加载网页
web_loader = WebBaseLoader(
    "https://python.langchain.com/docs/get_started/introduction"
)
# web_docs = web_loader.load()

# 5. 文档分割（重要！）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块大小
    chunk_overlap=50,    # 重叠部分（保持上下文连贯）
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)

# splits = text_splitter.split_documents(docs)
```

## 3. 向量存储与检索

### 3.1 使用ChromaDB

```python
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./chroma_db"
# )

# 相似度搜索
# results = vectorstore.similarity_search("Python是什么？", k=3)

# 最大边际相关性搜索（MMR）
# mmr_results = vectorstore.max_marginal_relevance_search("Python", k=3)
```

### 3.2 使用本地嵌入模型（省钱）

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=embeddings
# )
```

## 4. 完整RAG系统

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1. 自定义Prompt
prompt_template = """
使用以下上下文信息来回答问题。如果你不知道答案，就说不知道，不要编造答案。

上下文信息：
{context}

问题：{question}

答案：
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 2. 创建QA链
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}
# )

# 3. 查询
# result = qa_chain({"query": "什么是机器学习？"})
# print(result['result'])
```

## 5. 高级RAG技巧

### 5.1 混合检索

结合关键词检索（BM25）和向量检索，提高召回率。

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# bm25_retriever = BM25Retriever.from_documents(splits)
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.5, 0.5]
# )
```

### 5.2 重排序（Reranking）

使用Cross-Encoder对检索结果进行精排。

```python
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
# reranked_docs = reranker.compress_documents(docs, query)
```

### 5.3 多文档查询

跨多个知识库进行查询和结果聚合。

## 6. 实战案例：搭建本地知识库问答系统

这是一个可以复制即用的完整RAG脚本。它实现了从加载文档到回答问题的全流程。

**功能**：
1.  自动加载 `data/` 目录下的txt/pdf文件。
2.  使用 `text2vec-base-chinese` 进行嵌入（无需OpenAI Embeddings，节省成本）。
3.  使用 ChromaDB 存储向量。
4.  使用 OpenAI（或其他兼容接口）回答问题。

**准备工作**：
```bash
pip install langchain langchain-openai chromadb sentence-transformers pypdf
mkdir data
# 在data目录下放一些txt文件
```

**代码实现 (`simple_rag.py`)**：

```python
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 加载环境变量 (OPENAI_API_KEY)
load_dotenv()

class SimpleRAG:
    def __init__(self, doc_dir="./data", persist_dir="./chroma_db"):
        self.doc_dir = doc_dir
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )
        self.vectorstore = None
        self.qa_chain = None

    def build_knowledge_base(self):
        """构建知识库：加载 -> 分割 -> 向量化 -> 存储"""
        print(f"正在加载文档: {self.doc_dir}...")
        
        # 1. 加载文档
        loaders = {
            ".txt": DirectoryLoader(self.doc_dir, glob="**/*.txt", loader_cls=TextLoader),
            ".pdf": DirectoryLoader(self.doc_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        }
        
        documents = []
        for ext, loader in loaders.items():
            try:
                docs = loader.load()
                documents.extend(docs)
                print(f"加载了 {len(docs)} 个 {ext} 文档")
            except Exception as e:
                print(f"加载 {ext} 失败: {e}")

        if not documents:
            print("未找到文档，请检查data目录")
            return

        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        print(f"文档已分割为 {len(splits)} 个片段")

        # 3. 向量化并存储
        print("正在构建向量库 (可能需要几分钟)...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()
        print("知识库构建完成！")

    def load_knowledge_base(self):
        """加载已存在的知识库"""
        if os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print("已加载现有知识库")
        else:
            print("知识库不存在，请先构建")
            self.build_knowledge_base()

    def init_qa_chain(self):
        """初始化问答链"""
        if not self.vectorstore:
            self.load_knowledge_base()

        # 定义Prompt
        template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        
        上下文: {context}
        
        问题: {question}
        
        有用的回答:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 初始化LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0
        )

        # 构建链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

    def ask(self, query):
        """提问"""
        if not self.qa_chain:
            self.init_qa_chain()
            
        result = self.qa_chain({"query": query})
        
        print(f"\nQ: {query}")
        print(f"A: {result['result']}")
        print("\n[参考源]:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata.get('source', 'unknown')}: {doc.page_content[:50]}...")

# 使用示例
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # 首次运行需要构建
    # rag.build_knowledge_base()
    
    # 直接提问
    rag.ask("公司的请假政策是什么？")
```
