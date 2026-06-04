from typing import Annotated, TypedDict, Union
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class OverallState(TypedDict):
    """
    图全局状态
    """

    query: str
    """
    查询
    """
    messages: Annotated[list, add_messages]
    """
    消息列表
    """
    current_context: dict
    """
    当前内容
    """


class WeatherContext(BaseModel):
    """
    天气状态
    """

    date: str = Field(default="",description="查询日期，格式: YYYYMMDD，如果用户没有提及默认为空字符串")
    addr: str = Field(default="",description="查询的城市，如果用户没有提及默认为空字符串")
