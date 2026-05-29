"""
LLM 模块
提供 OpenAI 模型的创建和管理功能
"""
import dotenv 
dotenv.load_dotenv()
from langchain_openai import ChatOpenAI

def create_llm(model_name: str="Qwen/Qwen3-8B",temperature: float=0.0):
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


def create_embeding_model(model_name: str="Qwen/Qwen3-8B"):
    """
    创建嵌入模型用于将文本转换为向量表示
    
    返回值:
        嵌入模型实例
    """
    model = OpenAI(
        model=model_name,
        temperature=0.0,
        )
    return model

if __name__ == "__main__":
    llm = create_llm()
    print(llm.invoke("你知道我是谁么"))