from langchain_core.prompts import ChatPromptTemplate

# 意图识别提示
INTENT_PROMPT = ChatPromptTemplate.from_template(
    """
# Role
你是一个高精度的意图识别与槽位提取（Slot Filling）专家。你的任务是分析用户的输入，识别用户的真实意图，并提取出该意图所需的业务字段。

# 可用于逻辑处理配置表
- 今日日期：{date} 格式：YYYYMMDD 例如 20260604


# Context & Schema
- 当识别出对应意图时，你必须严格按照指定的 JSON 结构进行返回。
## 意图识别主体JSON
{router_info}

### 意图案例 1: 查询天气 (weather_query)
- **触发条件**：用户想要了解天气、气温、下雨等气象信息。
- **必需字段**：
  - `date`: 日期（例如：2026-06-04，从用户的提问中提取）
  - `addr`: 地址/位置（如：北京、上海市朝阳区）
- **意图主体的current_context字段格式**:
    {weather_json}

# 用户提问内容：
{query}
"""
)
