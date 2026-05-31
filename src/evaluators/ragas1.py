import os
from dotenv import load_dotenv

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    Faithfulness,
)

from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI

load_dotenv()

# =====================================================
# 1. Judge LLM
# =====================================================

from llm import create_llm

judge_llm = create_llm()

# Ragas 官方推荐包装方式
evaluator_llm = LangchainLLMWrapper(judge_llm)

# =====================================================
# 2. Dataset
# =====================================================

dataset = Dataset.from_dict(
    {
        "question": [
            "我买的衣服吊牌摘了，但有质量问题，能退吗？",
            "我的快递到哪了？给我单号！",
        ],
        "contexts": [
            [
                "售后政策：商品无吊牌不予退换货。",
                "特批条款：若商品存在出厂质量问题，即使吊牌摘除，经售后鉴定后亦可特批退款。",
            ],
            [
                "物流须知：我们合作的快递有顺丰和圆通。",
                "发货规则：每日16点前的订单当天发货。",
            ],
        ],
        "answer": [
            "质量问题即使摘了吊牌也可以申请退款。",
            "我们合作快递有顺丰和圆通。",
        ],
        "reference": [
            "因质量问题导致的吊牌摘除，可以特批退款。",
            "请提供订单号查询快递单号。",
        ],
    }
)

# =====================================================
# 3. Metrics
# =====================================================

metrics = [
    ContextRecall(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
]

print("Metrics Loaded:")

for m in metrics:
    print(type(m))

# =====================================================
# 4. Evaluate
# =====================================================

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
)

# =====================================================
# 5. Output
# =====================================================

print("\n================ RESULTS ================\n")
print(result)

df = result.to_pandas()

print(df)

df.to_csv(
    "ragas_result.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved -> ragas_result.csv")
