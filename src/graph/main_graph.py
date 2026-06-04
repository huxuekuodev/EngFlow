import asyncio

from langgraph.graph import StateGraph, START
from graph.nodes import route_by_intent, weather_tool
from graph.state import OverallState


async def main_graph():
    graph = StateGraph(OverallState)
    graph.add_node("route_by_intent", route_by_intent)
    graph.add_node("weather_query", weather_tool)

    graph.add_edge(START, "route_by_intent")
    app = graph.compile()
    result = await app.ainvoke({"query": "北京天气"})
    print(result)


if __name__ == "__main__":
    asyncio.run(main_graph())
