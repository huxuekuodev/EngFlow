import aiohttp
from langgraph.graph import END
from langgraph.types import interrupt, Command

from graph.state import OverallState, WeatherContext


async def weather_tool(state: OverallState) -> Command:
    query = state["current_context"]

    if not query["date"] or not query["addr"]:
        interrupt({"prompt": "请检查是否填写的必要信息*（日期和地址）"})
        # 再次进行意图识别
        return Command(goto="route_by_intent")

    # 查询天气
    print("对接查询天气")
    url = "https://restapi.amap.com/v3/weather/weatherInfo?key=7386463c09b9aec3d314200996353a3b&city=010"
    async with aiohttp.ClientSession() as session:
        # 2. 发送 GET 请求
        async with session.get(url) as response:
            # 3. 打印状态码
            print(f"状态码: {response.status}")

            # 4. 异步获取文本内容 (注意这里必须有 await)
            json_res = await response.json(encoding="utf-8")
            print(json_res)
    return Command(goto=END)
