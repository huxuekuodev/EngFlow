import os

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

from sqlalchemy import create_engine

from llm import OllamaEmbedding,create_llm
from prompts import DOC_TO_HYPO_PROMPT, QUERY_TO_HYPO_PROMPT
from vector import Vector, CustomPGDocStore


# 生成问题的结果模型
class HypoQuestionResult(BaseModel):
    Question: List[str] = Field(description="问题集合，每个元素是一个问题")


class HypoQuestionGenerator:
    """
    生成问题的类
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.doc_prompt = DOC_TO_HYPO_PROMPT
        self.query_prompt = QUERY_TO_HYPO_PROMPT

    def query_to_hypo(self, query: str) -> HypoQuestionResult:
        """
        处理查询，生成问题
        :param query: 查询字符串
        :return: 问题结果
        """
        chain = self.query_prompt | self.llm.with_structured_output(HypoQuestionResult)
        output_parser = PydanticOutputParser(pydantic_object=HypoQuestionResult)
        result = chain.invoke(
            {
                "doc": query,
                "structured_json": output_parser.get_format_instructions(),
            }
        )
        return result

    def query_to_hypo_list(self, query: str) -> List[str]:
        """
        处理查询，生成问题
        :param query: 查询字符串
        :return: 问题结果
        """
        chain = self.query_prompt | self.llm.with_structured_output(HypoQuestionResult)
        output_parser = PydanticOutputParser(pydantic_object=HypoQuestionResult)
        result = chain.invoke(
            {
                "user_query": query,
                "structured_json": output_parser.get_format_instructions(),
            }
        )
        return result.Question

    def doc_to_hypo_bath(self, doc: List[Document]) -> List[HypoQuestionResult]:
        """
        批量处理文档，生成问题
        :param doc: 文档列表
        :return: 问题列表
        """
        output_parser = PydanticOutputParser(pydantic_object=HypoQuestionResult)
        chain = (
            {
                "doc": RunnableLambda(lambda doc: doc.page_content),
                "structured_json": RunnableLambda(
                    lambda _: output_parser.get_format_instructions()
                ),
            }
            | self.doc_prompt
            | self.llm.with_structured_output(HypoQuestionResult)
        )
        result = chain.batch(doc)
        return result

if __name__ == "__main__":
    vector = Vector("engflow_collection", OllamaEmbedding())
    engine = create_engine(os.getenv("POSTGRE_URL", ""))
    pg_vector = CustomPGDocStore(engine, "engflow_parents_documents")
    retriever = MultiVectorRetriever(vectorstore=vector._milvus, docstore=pg_vector, doc_key="doc_id")
    llm = create_llm()
    HypoQuestionGenerator = HypoQuestionGenerator(llm)
    chain = RunnableLambda(lambda q:HypoQuestionGenerator.query_to_hypo_list(q)) | retriever.map()
    print(chain.invoke("你好"))