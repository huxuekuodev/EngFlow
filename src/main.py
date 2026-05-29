import uuid
from vector import MultiVector, Vector
from optimizer import HypoQuestionGenerator, SummaryOptimizer
from parser import CustomMDParser
from llm import create_llm
from vector import Vector
from langchain_classic.retrievers import MultiVectorRetriever


def merge_hypo_summary(
    hypo_result: List[HypoQuestionResult], summary_result: List[SummaryResult]
) -> List[Document]:
    pass


if __name__ == "__main__":
    parser = CustomMDParser("../../data/2026年海淀区高三二模英语阅读解析（C、D篇）.md")

    document_list = parser.parser()

    # 生成主键
    doc_key = "doc_id"
    doc_id_list = [str(uuid.uuid4()) for _ in range(len(document_list))]
    for i in range(len(document_list)):
        document_list[i].metadata[doc_key] = doc_id_list[i]

    llm = create_llm()
    # 生成假设问题
    hypo_generator = HypoQuestionGenerator(llm)
    hypo_result = hypo_generator.doc_to_hypo_bath(document_list)

    # 优化摘要
    summary_optimizer = SummaryOptimizer(llm)
    summary_result = summary_optimizer.batch_optimize(document_list)

    merged_list = merge_hypo_summary(hypo_result, summary_result)

    MultiVectorRetriever()
    pass
