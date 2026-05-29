import uuid
import json
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres.vectorstores import PGVector
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.stores import BaseStore
from llm import OllamaEmbedding, create_llm


# ==========================================
# 1. 基础配置（完全使用你的连接配置与表名）
# ==========================================
CONNECTION_STRING = (
    "postgresql+pg8000://huxuekuo:yiqizou89@82.156.200.4:5432/engflow_info"
)
CHILD_TABLE_NAME = "engflow_child_data"  # 对应你的子向量表
PARENT_TABLE_NAME = "engflow_parents_documents"  # 对应你的原始文档表

# 初始化你自己的模型
embeddings = OllamaEmbedding()
llmvm = create_llm()

# 初始化底层的 SQLAlchemy Engine 用来操作你的自定义文档表
engine = create_engine(CONNECTION_STRING)


# ==========================================
# 2. 自定义对接你的物理表 engflow_parents_documents
# ==========================================
class CustomPGDocStore(BaseStore[str, Document]):
    """适配你既有 PG 表结构 (doc_id, content, metadata) 的文档持久化存储器"""

    def __init__(self, engine, table_name):
        self.engine = engine
        self.table_name = table_name

    def yield_keys(self):
        with self.engine.connect() as conn:
            query = text(f"SELECT doc_id FROM {self.table_name}")
            for row in conn.execute(query):
                yield row[0]

    def mget(self, keys):
        if not keys:
            return []
        results = []
        with self.engine.connect() as conn:
            for key in keys:
                query = text(
                    f"SELECT content, metadata FROM {self.table_name} WHERE doc_id = :doc_id"
                )
                row = conn.execute(query, {"doc_id": key}).fetchone()
                if row:
                    # 读取出来后重新组装成 LangChain 认识的 Document 对象
                    meta = (
                        row[1]
                        if isinstance(row[1], dict)
                        else json.loads(row[1] or "{}")
                    )
                    results.append(Document(page_content=row[0], metadata=meta))
                else:
                    results.append(None)
        return results

    def mset(self, key_value_pairs):
        with self.engine.begin() as conn:
            for key, doc in key_value_pairs:
                query = text(
                    f"""
                    INSERT INTO {self.table_name} (doc_id, content, metadata) 
                    VALUES (:doc_id, :content, :metadata)
                    ON CONFLICT (doc_id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
                """
                )
                conn.execute(
                    query,
                    {
                        "doc_id": key,
                        "content": doc.page_content,
                        "metadata": json.dumps(doc.metadata),
                    },
                )

    def mdelete(self, keys):
        with self.engine.begin() as conn:
            for key in keys:
                query = text(f"DELETE FROM {self.table_name} WHERE doc_id = :doc_id")
                conn.execute(query, {"doc_id": key})


# 实例化你的自定义文档库（替代之前的 InMemoryStore）
docstore = CustomPGDocStore(engine, PARENT_TABLE_NAME)


# ==========================================
# 3. 初始化向量库与多向量检索器
# ==========================================
vectorstore = PGVector(
    connection=CONNECTION_STRING,
    embeddings=embeddings,
    collection_name=CHILD_TABLE_NAME,
    use_jsonb=True,
)

# 构造 MultiVectorRetriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,  # 此时这里已经是持久化的自定义 PG 存储了
    id_key="doc_id",  # 关联的 Key
)


# ==========================================
# 4. 核心逻辑：从 0 到 1 写入数据
# ==========================================
# 模拟你的原始长文档
raw_documents = [
    Document(
        page_content="LangChain是一个用于构建大语言模型应用的框架。它提供了组件化的接口，包括提示词管理、链式调用、向量库对接等。2026年，LangChain在生产环境中的生态更加成熟，特别是langchain_postgresql成为了PostgreSQL向量存储的首选方案。",
        metadata={"source": "tech_doc_01"},
    )
]

print("--- 开始处理文档 ---")

for doc in raw_documents:
    # A. 生成全局唯一的 doc_id 并存入文档库
    parent_id = str(uuid.uuid4())
    doc.metadata["doc_id"] = parent_id

    # 将原始文档存入你的物理表 engflow_parents_documents
    retriever.docstore.mset([(parent_id, doc)])
    print(f"-> 原始文档已成功存入 {PARENT_TABLE_NAME}，生成 doc_id: {parent_id}")

    # B. 利用 LLM 生成假设性问题 (Hypothetical Questions)
    chain_question = (
        ChatPromptTemplate.from_template(
            "根据以下文本，生成3个用户可能会问的简短假设性问题，每行一个：\n\n{doc}"
        )
        | llmvm
        | StrOutputParser()
    )
    response_q = chain_question.invoke({"doc": doc.page_content})
    questions = [q.strip() for q in response_q.split("\n") if q.strip()]

    # C. 利用 LLM 生成摘要 (Summary)
    chain_summary = (
        ChatPromptTemplate.from_template("请为以下文本生成一段精炼的摘要：\n\n{doc}")
        | llmvm
        | StrOutputParser()
    )
    summary = chain_summary.invoke({"doc": doc.page_content})

    print(f"生成的假设性问题:\n{questions}")
    print(f"生成的摘要:\n{summary}\n")

    # D. 将“问题”和“摘要”打包成 Document 对象，并绑定 parent_id
    sub_docs = []

    # 放入摘要
    sub_docs.append(
        Document(
            page_content=summary, metadata={"doc_id": parent_id, "type": "summary"}
        )
    )
    # 放入假设性问题
    for q in questions:
        sub_docs.append(
            Document(page_content=q, metadata={"doc_id": parent_id, "type": "question"})
        )

    # E. 将这些衍生文本及其向量，一并写入向量表 engflow_child_vectors
    retriever.vectorstore.add_documents(sub_docs)
    print(
        f"-> 已成功为该文档生成并向 {CHILD_TABLE_NAME} 写入了 {len(sub_docs)} 个衍生向量。"
    )

print("✅ 数据成功分块并写入 PostgreSQL 向量库与自定义文档库！\n")


# ==========================================
# 5. 检索测试
# ==========================================
print("--- 开始检索测试 ---")
query = "2026年用什么库连接 Postgres 向量库比较好？"
print(f"用户提问: {query}")

# 检索器会在内部：
# 1. 将 query 转化为向量
# 2. 在 engflow_child_vectors 中匹配“摘要”或“问题”
# 3. 拿到对应的 doc_id
# 4. 去 engflow_parents_documents 捞出原始长文本返回
retrieved_docs = retriever.invoke(query)

print(f"\n检索到的原始文档数量: {len(retrieved_docs)}")
if retrieved_docs:
    print(f"最终返回给 LLM 的原始文本内容:\n{retrieved_docs[0].page_content}")
    print(f"原始文本元数据: {retrieved_docs[0].metadata}")
