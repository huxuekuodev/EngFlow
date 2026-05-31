from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from llm import create_llm
from prompts import SUMMARY_INDEX_PROMPT
from parser import CustomMDParser


class SummaryResult(BaseModel):
    Question: List[str] = Field(description="摘要集合，每个元素是一个摘要")
    MetadataTag: List[str] = Field(
        description="为每个摘要添加元标签，每个元素是一个元标签，标签2-5个字概述内容，标签不能重复"
    )


class SummaryOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = SUMMARY_INDEX_PROMPT

    def optimize(self, text: str) -> SummaryResult:
        """
        对单个document进行优化,返回问题集合
        """
        structured_json = PydanticOutputParser(
            pydantic_object=SummaryResult
        ).get_format_instructions()
        chain = self.prompt | self.llm.with_structured_output(SummaryResult)
        response = chain.invoke({"doc": text, "structured_json": structured_json})
        return response

    # 批量处理documents
    def batch_optimize(self, documents: List[Document]) -> List[SummaryResult]:
        """
        批量处理documents,返回每个document的问题集合
        """
        structured_json = PydanticOutputParser(
            pydantic_object=SummaryResult
        ).get_format_instructions()
        chain = (
            {
                "doc": RunnableLambda(lambda doc: doc.page_content),
                "structured_json": RunnableLambda(lambda _: structured_json),
            }
            | self.prompt
            | self.llm.with_structured_output(SummaryResult)
        )
        results = chain.batch(documents)
        return results

    async def async_batch_optimize(
        self, documents: List[Document]
    ) -> List[SummaryResult]:
        """
        异步批量处理documents,返回每个document的问题集合
        """
        structured_json = PydanticOutputParser(
            pydantic_object=SummaryResult
        ).get_format_instructions()
        chain = (
            {
                "doc": RunnableLambda(lambda doc: doc.page_content),
                "structured_json": RunnableLambda(lambda _: structured_json),
            }
            | self.prompt
            | self.llm.with_structured_output(SummaryResult)
        )
        results = await chain.abatch(documents)
        return results


if __name__ == "__main__":
    parser = CustomMDParser(
        md_path="../../data/2026年海淀区高三二模英语阅读解析（C、D篇）.md",
        merge_titles=[
            ["阅读 D 篇", "阅读 C 篇", "逐选项定位汇总表"],
            ["【原文逐句对照与考点解析】", "【题目详细解析】"],
        ],
    )
    document_list = parser.parser()
    SummaryOptimizer = SummaryOptimizer(create_llm())
    response = SummaryOptimizer.batch_optimize(document_list)
    print(response)
