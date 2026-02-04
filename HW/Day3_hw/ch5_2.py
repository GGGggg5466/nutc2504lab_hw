import os
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL    = os.getenv("OPENAI_MODEL")

# =========================
# 1) Tool：get_weather
# =========================
@tool
def get_weather(city: str) -> str:
    """查詢城市天氣（示範版：用假資料）。輸入 city 回傳一句天氣描述。"""
    fake_db = {
        "台中": "台中晴天，氣溫 26 度",
        "台北": "台北下大雨，氣溫 18 度",
        "高雄": "高雄多雲，氣溫 28 度",
        "台南": "台南晴時多雲，氣溫 27 度",
    }
    city = city.strip()
    return fake_db.get(city, f"{city}：查無資料（示範版僅內建台中/台北/高雄/台南）")

tools = [get_weather]
tool_node_executor = ToolNode(tools)

# =========================
# 2) LLM（綁工具）
# =========================
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# =========================
# 3) State
# =========================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =========================
# 4) Node：agent（chatbot_node）
# =========================
def chatbot_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# =========================
# 5) Router：決定 tools 或 end
# =========================
def router(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        return "tools"
    else:
        return "end"

# =========================
# 6) 組裝 Graph
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "end": END
    }
)

# ✅ 循環邊：工具做完回到 agent 再生成最終回覆
workflow.add_edge("tools", "agent")

app = workflow.compile()

# =========================
# 7) stream 印出過程（像投影片）
# =========================
def print_event(event: dict):
    for node_name, state in event.items():
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)

        if node_name == "agent":
            if tool_calls:
                print(f"[AI 呼叫工具]: {tool_calls}")
            else:
                print(f"[AI]: {last.content}")
        elif node_name == "tools":
            print(f"[工具回傳]: {last.content}")

if __name__ == "__main__":
    print(app.get_graph().draw_ascii())
    print("\n=== ch5-2 weather assistant ===")
    print("輸入 exit 或 q 離開。\n")

    sys = SystemMessage(content=(
        "你是一個天氣查詢助理。"
        "當使用者問某城市天氣時，請呼叫工具 get_weather(city) 取得結果；"
        "拿到工具回傳後，用一句自然中文總結（可比較多城市）。"
    ))

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break

        for event in app.stream({"messages": [sys, HumanMessage(content=user_input)]}):
            print_event(event)
