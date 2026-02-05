import os
import json
import random
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://ws-05.huannago.com/v1")
API_KEY = os.getenv("API_KEY", "aabbccdd1115")
MODEL = os.getenv("MODEL", "Qwen3-VL-8B-Instruct-BF16.gguf")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)

SYSTEM_PROMPT = """你是一個天氣助理。
如果使用者問天氣，請呼叫工具 get_weather(city)。
城市請從使用者句子中抓取（台北/台中/高雄）。
如果看不出城市，就先反問使用者要查哪個城市。
"""

# =============================
# Tool (故意隨機失敗)
# =============================
@tool
def get_weather(city: str) -> str:
    """查詢指定城市的天氣。"""
    # 故意 50% 失敗
    if random.random() < 0.5:
        return "系統錯誤：天氣資料庫連線失敗，請再試一次。"

    if "台北" in city:
        return "台北大雨，氣溫 18 度"
    elif "台中" in city:
        return "台中晴天，氣溫 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

# =============================
# State
# =============================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =============================
# Nodes
# =============================
def chatbot_node(state: AgentState) -> AgentState:
    resp = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [resp]}

tool_node_executor = ToolNode(tools)

def fallback_node(state: AgentState) -> AgentState:
    """
    超過最大重試次數時：
    一定要回 ToolMessage 並帶 tool_call_id，
    否則 tool call 會懸空，流程會卡住。
    """
    last = state["messages"][-1] 
    tool_call_id = last.tool_calls[0]["id"] if getattr(last, "tool_calls", None) else "unknown"

    msg = ToolMessage(
        content="系統提示：已超過最大重試次數（Max Retries Reached），請稍後再試。",
        tool_call_id=tool_call_id,
    )
    return {"messages": [msg]}

# =============================
# Router (核心：數 retry 次數)
# =============================
def router(state: AgentState) -> Literal["tools", "fallback", "end"]:
    messages = state["messages"]
    last = messages[-1]

    # 1) 如果沒有 tool_calls => 結束（代表 LLM 直接回答或反問）
    if not getattr(last, "tool_calls", None):
        return "end"

    # 2) 計算「連續失敗」次數：往回找 ToolMessage
    retry_count = 0

    # 從倒數第二個開始找（因為最後一個是 AIMessage toolcall）
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            if "系統錯誤" in msg.content:
                retry_count += 1
            else:
                # 一旦出現成功的 tool 回覆，就停止計數（連續失敗被中斷）
                break
        elif isinstance(msg, HumanMessage):
            break

    print(f"DEBUG: 目前連續重試次數：{retry_count}")

    # 3) 設定上限（老師截圖是 >= 3 走 fallback）
    if retry_count >= 3:
        return "fallback"

    return "tools"

# =============================
# Build Graph
# =============================
def build_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", chatbot_node)
    workflow.add_node("tools", tool_node_executor)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "fallback": "fallback",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("fallback", "agent")  

    return workflow.compile()

# =============================
# CLI
# =============================
if __name__ == "__main__":
    app = build_app()
    print(app.get_graph().draw_ascii())

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "q", "quit"):
            break

        events = app.stream({"messages": [HumanMessage(content=user_input)]})
        for event in events:
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if getattr(msg, "tool_calls", None):
                        print("-> [Agent]: 決定呼叫工具（重試中...）")
                    else:
                        print(f"-> [Agent]: {msg.content}")
                elif key == "tools":
                    tool_msg = value["messages"][-1]
                    if "系統錯誤" in tool_msg.content:
                        print("-> [Tools]: 系統錯誤...（失敗）")
                    else:
                        print(f"-> [Tools]: {tool_msg.content}")
                elif key == "fallback":
                    print("-> [Fallback]: ⚠️ 觸發 fallback，停止重試")
