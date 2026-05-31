import uuid
import asyncio
import os
from parser import CustomMDParser
from llm import create_llm
from optimizer import HypoQuestionGenerator, SummaryOptimizer, ParentChildOptimizer
from optimizer.base import merge_hypo_summary
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def run():
    parser = CustomMDParser(
        "./data/2026年海淀区高三二模英语阅读解析（C、D篇）.md",
        merge_titles=[
            ["阅读 D 篇", "阅读 C 篇", "逐选项定位汇总表"],
            ["【原文逐句对照与考点解析】", "【题目详细解析】"],
        ],
    )

    document_list = parser.parser()

    # 生成主键
    doc_key = "doc_id"
    doc_id_list = [str(uuid.uuid4()) for _ in range(len(document_list))]
    for i in range(len(document_list)):
        document_list[i].metadata[doc_key] = doc_id_list[i]

    llm = create_llm()
    # 生成假设问题
    print(f"批量生成假设性问题开始")
    hypo_generator = HypoQuestionGenerator(llm)
    summary_optimizer = SummaryOptimizer(llm)
    childSpliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    parentsChild = ParentChildOptimizer(document_list, childSpliter)
    results = await asyncio.gather(
        hypo_generator.async_doc_to_hypo_bath(document_list),
        summary_optimizer.async_batch_optimize(document_list),
        parentsChild.async_get_child_documents(),
        return_exceptions=True,
    )
    merged_list = merge_hypo_summary(results[0], results[1], results[2], doc_id_list)

    print(f"合并后的文档{len(merged_list)}")


if __name__ == "__main__":

    asyncio.run(run())
