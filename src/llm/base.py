"""
LLM 模块
提供 OpenAI 模型的创建和管理功能
"""

import dotenv
import os

dotenv.load_dotenv()
from langchain_openai import ChatOpenAI, OpenAI
from typing import List
from langchain_core.embeddings import Embeddings  # 👈 必须导入基类
from ollama import Client  # 确保你安装了 ollama 库


def create_llm(model_name: str = "Qwen/Qwen3-8B", temperature: float = 0.0):
    """
    创建 LLM 模型用于生成文本

    返回值:
        LLM 模型实例

    参数:
        model_name: 模型名称
        temperature: 温度参数，控制生成文本的随机性
    """
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )
    return model


def create_openai(model: str = "Qwen/Qwen3-8B"):
    """
    创建 OpenAI 模型实例

    参数:
        model: 模型名称，默认为空字符串，表示使用默认模型

    返回值:
        OpenAI 模型实例
    """
    return OpenAI(model=model)


class OllamaEmbedding(Embeddings):

    def __init__(self, model_name: str = "qwen3-embedding:4b", dimensions: int = 1024):
        """
        初始化 Ollama 嵌入模型
        :param model_name: 默认使用的嵌入模型名称
        :param dimensions: 嵌入向量的维度（Qwen3 默认支持多维度，1024 是常见高性能选择）
        """
        # 从环境变量获取 URL，如果未设置则默认连接本地
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.client = Client(host=ollama_url)
        self.model_name = model_name
        self.dimensions = dimensions

    def get_embeddings(self, inpt: List[str]) -> List[List[float]]:
        """
        底层的批量获取嵌入向量方法
        """
        try:
            emb_resp = self.client.embed(
                model=self.model_name, dimensions=self.dimensions, input=inpt
            )
            return emb_resp.embeddings
        except Exception as e:
            # 生产环境增加异常捕获
            raise RuntimeError(f"Ollama 嵌入服务调用失败: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """
        实现 LangChain 接口：为单个查询（提问）生成向量
        """
        # 即使是单个文本，Ollama 接口也需要包裹成 list 传进去，拿到结果后取第 0 个
        embeddings = self.get_embeddings([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        实现 LangChain 接口：为批量文档生成向量
        """
        if not texts:
            return []
        return self.get_embeddings(texts)


def create_rerank_llm(
    model_name: str = "BAAI/bge-reranker-v2-m3", temperature: float = 0.0
):
    model = ChatOpenAI(
        model=model_name,
        base_url="https://api.siliconflow.cn/v1/rerank",
        temperature=temperature,
    )
    return model


if __name__ == "__main__":
    llm = create_rerank_llm()
    print(llm.invoke("你知道我是谁么"))
