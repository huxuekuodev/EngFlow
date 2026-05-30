from typing import List, Optional, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction, MilvusCollectionHybridSearchRetriever
# 加载配置文件
load_dotenv()

# 双向量检索
class MultiVector:
    """
        初始化介绍：
            默认索引：HNSW 图索引 BM25 稀疏索引
        搜索介绍：e
            Langchain_milvus 在搜索时会默认使用稀疏以及密集索引两种方式
    """
    # * 后面必须是实名参数
    def __init__(self,collection_name,
                 # 使用的向量模型
                 embed,
                 *,
                 # 创建表是 索引配置参数
                 index_params:Optional[Union[dict, List[dict]]]=None,
                 # 搜索参数
                 search_params:Optional[Union[dict, List[dict]]] = None):
        self._collection_name = collection_name
        if index_params is None:
            index_params = self._default_index_params()

        if search_params is None:
            search_params = self._default_search_params()

        self._milvus = Milvus(embedding_function=embed,
                              builtin_function=BM25BuiltInFunction(
                                input_field_names="text",  # 原始文本输入字段（langchain 默认叫 text）
                                output_field_names="sparse"  # 对应的稀疏向量输出字段
                            ),

                            connection_args={"uri": "http://localhost:19530","alias": "default","token":"root:Milvus"},
                            auto_id=True,
                            drop_old=True,
                            collection_name=collection_name,
                            vector_field=["vector","sparse"],
                            index_params=index_params,
                            search_params=search_params,
                            enable_dynamic_field=True)

    def _default_search_params(self):
        """
        生产级 双索引 搜索参数
        对应：HNSW稠密向量 + BM25稀疏向量
        低延迟、高召回、高并发稳定
        """
        return {
            # 1. HNSW 稠密向量搜索参数
            "vector": {
                "metric_type": "COSINE",
                "params": {
                    "ef": 128,  # 生产级：128~256，高QPS用128，高召回用256
                    "offset": 0 # 是否跳过 offset 条 后开始拿取结果
                }
            },
            # 2. BM25 稀疏向量搜索参数
            "sparse": {
                "metric_type": "IP",
                "params": {
                    "bm25_k1": 1.2, # BM25 标准参数
                    "bm25_b": 0.75, # 全文检索最优
                    "offset": 0
                }
            }
        }

    def _default_index_params(self):
        return [
            # 索引1：稠密向量索引 HNSW（M=3）
            {
                "field_name": "vector",
                "index_name": "vector_idx",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 3,  # 你要求的 M=3
                    "efConstruction": 300,  # 生产级：200~300（越高越准，构建越慢）
                    "ef": 128  # 搜索时保留的候选列表，生产 128~256
                }
            },
            # 索引2：稀疏向量索引 BM25
            {
                "field_name": "sparse",
                "index_name": "sparse_idx",
                "index_type": "SPARSE_INVERTED_INDEX", # BM25 固定配置， 只有这一个
                "metric_type": "BM25", #  BM25 固定配置， 只有这一个
                "params": {
                    "bm25_k1": 1.2,  # BM25 标准参数
                    "bm25_b": 0.75  # 全文检索最优
                },
                "analyzer":{
                    "type" : "jieba"
                }
            }
        ]

    # 添加文档
    def add_document(self,docs:List[Document]):
        self._milvus.add_documents(docs)

    def hy_search(self,query:str,k:int=5,param: dict=None,expr:str=None,weights:List =None):
        """
            混合检索： langchain_milvus 默认使用稀疏向量和密集向量同时检索
                排序是使用： RRFRanker 方式排序
        :param query:
        :param k:
        :param param:
        :param expr:
        :param weights: 设置向量权重[0.7,0.3] 密集向量权重70，稀疏向量权重30
        :return:
        """
        if weights:
            return self._milvus.similarity_search(query=query,k=k,param=param,expr=expr,
                                                  ranker_type="weights",ranker_params={"weights":weights})
        else:
            return self._milvus.similarity_search(query=query, k=k, param=param, expr=expr)

    def get_retriever(self,search_type:str="similarity",k = 5,weights=None):
        """
            返回检索器
        :param search_type: mmr 和 similarity_score_threshold 不支持多向量检索
        :param k:
        :param weights:
        :return:
        """
        return self._milvus.as_retriever(
            search_type=search_type,  # 搜索类型
            search_kwargs={  # 搜索参数
                "k": k,  # 返回几条
                "ranker_type": "rrf",  # 结果融合方式
                "ranker_params": {"k":100},
                "filter":{"category":"NarrativeText"} # 过滤元数据

                # "ranker_type": "weighted",  # <--- 修改这里：指定为加权融合
                # "ranker_params": {
                #     "weights": [0.6, 0.4]  # <--- 修改这里：设置权重
                    # 注意：列表的顺序必须对应你在 Collection 中定义的向量字段顺序
                    # 通常是 [稠密向量权重, 稀疏向量权重]
                # },
            }
        )


# 单向量检索
class Vector:

    def __init__(self,collection_name:str,embed):
        self._milvus = Milvus(embedding_function=embed,
                              connection_args={"uri": "http://localhost:19530", "alias": "default","token": "root:Milvus"},
                              auto_id=True,
                              drop_old=False,
                              collection_name=collection_name,
                              vector_field=["vector"],
                              index_params=self.__index_params(),
                              enable_dynamic_field=True)

    def __index_params(self):
        return [{
            "field_name": "vector",
            "index_name": "vector_idx",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "ef": 128, "offset": 0
            }
        }]

# if __name__ == "__main__":
#     logger = logger.get_logger()
#     embed = OllamaEmbedding()
#     milvus = BaseMilvus("test_collection",embed=embed)
#     pdfparser = PDFParser()
#     listdoc = pdfparser.load_to_file("../../data/2026年昌平区高三二模英语阅读解析（C、D篇）.pdf")
#     milvus.add_document(listdoc)
    # print(milvus.get_retriever().invoke("权衡它"))