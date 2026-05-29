from typing import List, Optional
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class CustomMDParser:
    def __init__(self, md_path: str, merge_titles: List[List[str]]):
        self.md_path = md_path
        self.merge_titles = merge_titles

    def parser(self) -> List[Document]:
        loader = UnstructuredMarkdownLoader(file_path=self.md_path, mode="elements", strategy="hi_res")
        return self.__process_documents(loader.load())

    def __process_documents(self, documents: list[Document]) -> List[Document]:
        file_name = documents[0].page_content
        level1_titles = set(self.merge_titles[0])
        level2_titles = set(self.merge_titles[1]) if len(self.merge_titles) > 1 else set()

        results, docs = [], []
        l1_title, l1_content = None, ""

        for doc in documents[1:]:
            content = doc.page_content.strip()

            if content in level1_titles:
                if docs:
                    results.append(self._create_doc(file_name, docs, l1_title, l1_content))
                l1_title, l1_content = content, ""
                docs = []

            elif content in level2_titles:
                if docs:
                    results.append(self._create_doc(file_name, docs, l1_title, l1_content))
                docs = [content]

            elif docs:
                docs.append(content)
            else:
                l1_content += ("\n\n" + content if l1_content else content)

        if docs:
            results.append(self._create_doc(file_name, docs, l1_title, l1_content))

        return results

    def _create_doc(self, file_name: str, docs: list, l1_title: Optional[str], l1_content: str) -> Document:
        text = "\n\n".join(docs)
        if l1_title and l1_content:
            text = f"{l1_title}\n\n{l1_content}\n\n{text}"
        elif l1_title:
            text = f"{l1_title}\n\n{text}"
        return Document(page_content=text, metadata={"file_name": file_name, "title": docs[0]})


if __name__ == "__main__":
    parser = CustomMDParser(
        md_path="../data/2026年海淀区高三二模英语阅读解析（C、D篇）.md",
        merge_titles=[
            ["阅读 D 篇", "阅读 C 篇", "逐选项定位汇总表"],
            ["【原文逐句对照与考点解析】", "【题目详细解析】"],
        ],
    )
    results = parser.parser()
    print(f"总文档数: {len(results)}")

