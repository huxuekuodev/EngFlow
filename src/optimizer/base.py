from .hypo_question_generator import HypoQuestionResult
from .summary_optimizer import SummaryResult
from typing import List
from langchain_core.documents import Document


from .hypo_question_generator import HypoQuestionResult
from .summary_optimizer import SummaryResult
from typing import List, Union
from langchain_core.documents import Document


def merge_hypo_summary(
    hypo_result: Union[List[HypoQuestionResult], Exception],
    summary_result: Union[List[SummaryResult], Exception],
    child_document_list: Union[List[List[Document]], Exception],
    doc_ids: List[str],
) -> List[Document]:
    """
    通过索引动态合并。即使某个任务整体失败（Exception），也不影响其他任务的数据合并。
    """
    document_list = []

    # 检查三大任务本身是否整体崩溃
    hypo_ok = isinstance(hypo_result, list)
    summary_ok = isinstance(summary_result, list)
    child_ok = isinstance(child_document_list, list)

    if not hypo_ok:
        print(f"通知: 合并时跳过假设问题，因为生成器报错: {hypo_result}")
    if not summary_ok:
        print(f"通知: 合并时跳过摘要优化，因为优化器报错: {summary_result}")
    if not child_ok:
        print(f"通知: 合并时跳过父子块切分，因为切分器报错: {child_document_list}")

    # 以唯一的绝对基准 doc_ids 的索引进行遍历
    for idx, doc_id in enumerate(doc_ids):

        # 1. 动态提取并合并假设问题
        if hypo_ok and idx < len(hypo_result):
            hypo = hypo_result[idx]
            if hypo and hasattr(hypo, "Question") and hypo.Question:
                document_list.extend(
                    [
                        Document(page_content=q, metadata={"doc_id": doc_id})
                        for q in hypo.Question
                    ]
                )

        # 2. 动态提取并合并摘要
        if summary_ok and idx < len(summary_result):
            summary = summary_result[idx]
            if summary and hasattr(summary, "Question") and summary.Question:
                document_list.extend(
                    [
                        Document(page_content=q, metadata={"doc_id": doc_id})
                        for q in summary.Question
                    ]
                )

        # 3. 动态提取并合并子文档块
        if child_ok and idx < len(child_document_list):
            children = child_document_list[idx]
            if children:
                document_list.extend(
                    [
                        Document(
                            page_content=d.page_content, metadata={"doc_id": doc_id}
                        )
                        for d in children
                    ]
                )

    print(f"合并完成，最终成功组装片段总数: {len(document_list)}")
    return document_list
