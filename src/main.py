import uuid
import  os
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine
from optimizer import HypoQuestionGenerator, SummaryOptimizer, ParentChildOptimizer
from optimizer.hypo_question_generator import HypoQuestionResult
from optimizer.summary_optimizer import SummaryResult
from parser import CustomMDParser
from llm import create_llm, OllamaEmbedding
from vector import Vector,CustomPGDocStore
from langchain_classic.retrievers import MultiVectorRetriever


def merge_hypo_summary(hypo_result:List[HypoQuestionResult],
                       summary_result:List[SummaryResult],
                       child_document_list:List[List[Document]],
                       doc_ids:List[str]) -> List[Document]:
    """
        摘要内容、假设问题,父子块合并
    :param hypo_result:
    :param summary_result:
    :param doc_ids:
    :return:
    """
    print(f"假设性问题长度：{len(hypo_result)}, 摘要问题长度：{len(summary_result)}，父子块长度：{len(child_document_list)}主键ID长度：{len(doc_ids)}")
    document_list = []
    for hypo, summary,children,id in zip(hypo_result, summary_result, child_document_list, doc_ids):
        document_list.extend([Document(page_content=q,metadata={"doc_id":id})for q in hypo.Question])
        document_list.extend([Document(page_content=q, metadata={"doc_id": id}) for q in summary.Question])
        document_list.extend(Document(page_content=d.page_content,metadata={"doc_id": id}) for d in children)
    print(f"合并后的长度{len(document_list)}")
    return document_list

# Document 去重
def unique_documents_by_RRF(doc_list:List[List[Document]],k:int=60) -> List[Document]:
    pass

if __name__ == "__main__":
    # parser = CustomMDParser("../data/2026年海淀区高三二模英语阅读解析（C、D篇）.md",    merge_titles=[
    #         ["阅读 D 篇", "阅读 C 篇", "逐选项定位汇总表"],
    #         ["【原文逐句对照与考点解析】", "【题目详细解析】"],
    #     ],)
    #
    # document_list = parser.parser()
    #
    # # 生成主键
    # doc_key = "doc_id"
    # doc_id_list = [str(uuid.uuid4()) for _ in range(len(document_list))]
    # for i in range(len(document_list)):
    #     document_list[i].metadata[doc_key] = doc_id_list[i]
    #
    llm = create_llm()
    # # 生成假设问题
    # print(f"批量生成假设性问题开始")
    # hypo_generator = HypoQuestionGenerator(llm)
    # hypo_result = hypo_generator.doc_to_hypo_bath(document_list)
    #
    # # 优化摘要
    # print(f"批量摘要开始")
    # summary_optimizer = SummaryOptimizer(llm)
    # summary_result = summary_optimizer.batch_optimize(document_list)
    #
    # print(f"父子块拆分开始")
    # childSpliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    # parentsChild = ParentChildOptimizer(document_list,childSpliter)
    # child_documents = parentsChild.get_child_documents()
    #
    # merged_list = merge_hypo_summary(hypo_result, summary_result,child_documents,doc_id_list)

    vector = Vector("engflow_collection",OllamaEmbedding())
    engine = create_engine(os.getenv("POSTGRE_URL",""))
    pg_vector = CustomPGDocStore(engine,"engflow_parents_documents")
    retriever = MultiVectorRetriever(vectorstore=vector._milvus,docstore=pg_vector,doc_key="doc_id")
    # 添加文档
    # retriever.vectorstore.add_documents(merged_list)
    # retriever.docstore.mset(list(zip(doc_id_list,document_list)))

    HypoQuestionGenerator = HypoQuestionGenerator(llm)
    chain = RunnableLambda(lambda q: HypoQuestionGenerator.query_to_hypo_list(q)) | retriever.map()

    print(chain.invoke("关于动词原型考点"))
    pass
