import asyncio
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from langgraph.graph import StateGraph, START
from graph.nodes import route_by_intent, weather_tool
from graph.state import OverallState


async def main_graph():
    DB_URI = "postgresql://huxuekuo:yiqizou89@82.156.200.4:5432/postgres"
    async with (
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # 首次运行时需要初始化数据库表
        await store.setup()
        await checkpointer.setup()

        graph = StateGraph(OverallState)
        graph.add_node("route_by_intent", route_by_intent)
        graph.add_node("weather_query", weather_tool)
        graph.add_edge(START, "route_by_intent")
        app = graph.compile(checkpointer=checkpointer, store=store)
        result = await app.ainvoke(
            {"query": "天气"}, config={"configurable": {"thread_id": "1"}}
        )
        print(f"result: {result}")
        result = await app.ainvoke(
            Command(resume="北京后天的天气"),
            config={"configurable": {"thread_id": "1"}},
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main_graph())
