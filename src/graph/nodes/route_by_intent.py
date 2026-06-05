from datetime import datetime
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langgraph.graph import END
from graph.state import OverallState, WeatherContext
from langgraph.types import Command
from llm import create_llm
from prompts import INTENT_PROMPT


class IntentEnum(str, Enum):
    WEATHER = "weather_query"  # 查询天气
    TRAVEL = "travel_plan"  # 旅游出行


class RouteInfo(BaseModel):
    """
    路由信息
    """

    intent: Optional[IntentEnum] = Field(
        default=None,
        description="意图识别分类结果: weather_query(查天气), travel_plan(旅游出行)",
    )
    current_context: Optional[dict] = Field(
        default=None,
        description="结合意图识别类型，返回不同字段信息",
    )


async def route_by_intent(state: OverallState) -> Command[OverallState]:
    """
    路由根据意图
    """
    llm = create_llm()
    chain = INTENT_PROMPT | llm.with_structured_output(RouteInfo)
    output_parser = PydanticOutputParser(pydantic_object=RouteInfo)
    weather_parser = PydanticOutputParser(pydantic_object=WeatherContext)
    intent_info = await chain.ainvoke(
        {
            "router_info": output_parser.get_format_instructions(),
            "weather_json": weather_parser.get_format_instructions(),
            "query": state["query"],
            "date": datetime.now().strftime("%Y%m%d"),
        }
    )
    print(intent_info)
    intent = intent_info.intent
    if intent == IntentEnum.WEATHER:
        return Command(
            update={
                "current_context": intent_info.current_context,
                "messages": "查询天气",
            },
            goto=intent.value,
        )
    # 如果没有意图，直接结束
    return Command(update={"messages": "意图识别失败，请重新输入"}, goto=END)
