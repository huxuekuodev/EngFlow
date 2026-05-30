from .hypo_question_generator import HypoQuestionResult
from .summary_optimizer import SummaryResult
from typing import List
from langchain_core.documents import Document


def merge_hypo_summary(
    hypo_result: List[HypoQuestionResult],
    summary_result: List[SummaryResult],
    child_document_list: List[List[Document]],
    doc_ids: List[str],
) -> List[Document]:
    """
        摘要内容、假设问题,父子块合并
    :param hypo_result:
    :param summary_result:
    :param doc_ids:
    :return:
    """
    print(
        f"假设性问题长度：{len(hypo_result)}, 摘要问题长度：{len(summary_result)}，父子块长度：{len(child_document_list)}主键ID长度：{len(doc_ids)}"
    )
    document_list = []
    for hypo, summary, children, id in zip(
        hypo_result, summary_result, child_document_list, doc_ids
    ):
        document_list.extend(
            [Document(page_content=q, metadata={"doc_id": id}) for q in hypo.Question]
        )
        document_list.extend(
            [
                Document(page_content=q, metadata={"doc_id": id})
                for q in summary.Question
            ]
        )
        document_list.extend(
            Document(page_content=d.page_content, metadata={"doc_id": id})
            for d in children
        )
    print(f"合并后的长度{len(document_list)}")
    return document_list
