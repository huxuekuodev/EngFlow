from dataclasses import dataclass
import json
from sqlalchemy import create_engine, text, Engine
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

# ==========================================
# 1. 基础配置（完全使用你的连接配置与表名）
# ==========================================
CONNECTION_STRING = "postgresql+pg8000://huxuekuo:yiqizou89@82.156.200.4:5432/postgres"
PARENT_TABLE_NAME = "engflow_parents_documents"  # 对应你的原始文档表
# 初始化底层的 SQLAlchemy Engine 用来操作你的自定义文档表
engine = create_engine(CONNECTION_STRING)


# ==========================================
# 2. 自定义对接你的物理表 engflow_parents_documents
# ==========================================
@dataclass
class CustomPGDocStore(BaseStore[str, Document]):
    """适配你既有 PG 表结构 (doc_id, content, metadata) 的文档持久化存储器"""

    engine: Engine
    table_name: str

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


print("自定义文档库已初始化，准备进行测试...")
# 实例化你的自定义文档库（替代之前的 InMemoryStore）
docstore = CustomPGDocStore(engine, PARENT_TABLE_NAME)
print("正在测试 mset 方法...")
docstore.mset(
    [(1, Document(page_content="这是一个测试文档", metadata={"source": "测试"}))],
)
