# from langchain_cohere import CohereRerank
# from langchain_classic.retrievers import ContextualCompressionRetriever
#
# # 初始化 Cohere 的 Rerank 压缩器
# compressor = CohereRerank(model="rerank-multilingual-v3.0", cohere_api_key="your-api-key")
#
# # 构建双阶段检索器
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=base_retriever
# )