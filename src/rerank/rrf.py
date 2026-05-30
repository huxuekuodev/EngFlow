# defaultdict 当访问一个不存在的键时，会自动创建一个默认值并返回它，而不是抛出 KeyError 异常。
from collections import defaultdict
from typing import List
from langchain_core.documents import Document


def rrf_rerank(results: List[List[Document]], k=60):
    """
    RRF (Reciprocal Rank Fusion) 是一种简单而有效的融合方法，用于将多个检索结果列表合并成一个综合排名列表。
    RRF 的核心思想是根据每个文档在不同结果列表中的排名来计算一个综合得分，排名越靠前的文档得分越高。
    参数:
    - results: 一个包含多个检索结果列表的列表，每个结果列表是一个包含文档ID和得分的元组列表。
    - k: 一个常数，用于调整排名的影响，默认值为60。
    返回:
    - 一个包含融合得分的列表，每个元素是一个包含文档ID和融合得分的元组，按融合得分降序排列。
    """
    # Initialize a dictionary to store the fused scores
    fused_scores = defaultdict(float)

    # Iterate over each result list
    for result in results:
        for rank, item in enumerate(result):
            doc_id = item["metadata"]["doc_id"]
            # Calculate the reciprocal rank score and add it to the fused score
            fused_scores[doc_id] += 1 / (rank + k)

    # Convert the dictionary to a list of tuples and sort by fused score
    fused_scores_list = [(doc_id, score) for doc_id, score in fused_scores.items()]
    fused_scores_list.sort(key=lambda x: x[1], reverse=True)

    return fused_scores_list
