from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ParentChildOptimizer:

    def __init__(self, parents:List[Document],child:RecursiveCharacterTextSplitter):
        self.__parents = parents
        self.__child = child

    def get_child_documents(self) -> List[List[Document]]:
        """
            拆分子块
        :return:
        """
        documents = []
        for p in self.__parents:
            documents.append([Document(page_content=s) for s in self.__child.split_text(p.page_content)])
        return documents


if __name__ == "__main__":
    parentsChild = ParentChildOptimizer([Document(page_content="123333333333333333333333333")],RecursiveCharacterTextSplitter(chunk_size=4,chunk_overlap=1))
    print(parentsChild.get_child_documents())