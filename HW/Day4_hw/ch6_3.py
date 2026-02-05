# ch6_3_human_review.py
# 實作：LangGraph 人工審核的訂單資訊（Human-in-the-loop）
# - Agent 抽訂單 → Tool 回 JSON → 若 VIP 觸發人工審核 → 再回 Agent 收尾

import os
import json
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


# =============================
# 0) 環境設定 / LLM
# =============================
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://ws-02.wade0426.me/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "gemma-3-27b-it")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)

SYSTEM_PROMPT = """你是一個訂單助理。
使用者會用自然語言輸入訂單資訊（姓名、電話、商品、數量、地址）。
你的任務是呼叫工具 extract_order_data 來抽取訂單欄位。
如果資訊缺少，請先追問缺少的欄位，不要亂猜。
如果資訊足夠，請呼叫工具並帶入正確參數。
"""


# =============================
# 1) VIP 名單
# =============================
VIP_LIST = ["AI哥", "一點馬"]


# =============================
# 2) Tool：抽取訂單資料（回 dict）
# =============================
@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具：
    專門用於從結構化參數中組出訂單資訊（姓名、電話、商品、數量、地址）。
    """
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address,
    }


tools = [extract_order_data]
llm_with_tools = llm.bind_tools(tools)
tool_node_executor = ToolNode(tools)


# =============================
# 3) State
# =============================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =============================
# 4) Nodes
# =============================
def agent_node(state: AgentState) -> AgentState:
    """思考節點：決定要不要呼叫工具，或直接回覆/追問"""
    resp = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [resp]}


def human_review_node(state: AgentState) -> AgentState:
    """
    人工審核節點：
    當偵測到 VIP 客戶時，流程會卡在這裡等待管理員輸入 ok / 其他拒絕。
    """
    print("\n" + "=" * 30)
    print("🚨 觸發人工審核機制：偵測到 VIP 客戶！")
    print("=" * 30)

    last_msg = state["messages"][-1]
    # last_msg 通常是 ToolMessage（工具回傳 JSON）
    print(f"待審核資料：{getattr(last_msg, 'content', str(last_msg))}")

    review = input(">>> 管理員請指示（輸入 'ok' 通過，其他則拒絕）：").strip().lower()

    if review == "ok":
        # 讓 agent 收到一個「人類已通過」的訊息，回去收尾/確認訂單
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "【系統公告】管理員已人工審核通過此 VIP 訂單。"
                        "請你向使用者確認訂單資料是否正確，並告知已開始處理。"
                    )
                )
            ]
        }

    # 拒絕：同樣回一段訊息，讓 agent 來對使用者說明拒絕/取消
    return {
        "messages": [
            HumanMessage(
                content=(
                    "【系統公告】管理員拒絕此 VIP 訂單。"
                    "請你向使用者致歉並告知訂單已取消，必要時請他重新下單或補齊資訊。"
                )
            )
        ]
    }


# =============================
# 5) Routers（Edges）
# =============================
def entry_router(state: AgentState):
    """
    判斷 Agent 是否要呼叫工具：
    - 如果最後一則 AIMessage 有 tool_calls -> tools
    - 否則 -> END
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def post_tool_router(state: AgentState) -> Literal["human_review", "agent"]:
    """
    工具執行完後的路由：
    - 解析 ToolMessage 的 JSON
    - name 在 VIP_LIST -> human_review
    - 否則 -> agent（回去收尾/確認）
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        try:
            data = json.loads(last_message.content)
            user_name = data.get("name", "")
            if user_name in VIP_LIST:
                print(f"DEBUG: 發現 VIP [{user_name}] -> 轉向人工審核")
                return "human_review"
        except Exception as e:
            print(f"DEBUG: ToolMessage JSON 解析失敗：{e}")

    # 非 VIP 或解析失敗，都回 agent 讓它處理後續（可追問/重新抽取）
    return "agent"


# =============================
# 6) Build Graph
# =============================
def build_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_executor)
    workflow.add_node("human_review", human_review_node)

    workflow.set_entry_point("agent")

    # (1) agent -> tools 或 END
    workflow.add_conditional_edges(
        "agent",
        entry_router,
        {"tools": "tools", END: END},
    )

    # (2) tools -> human_review 或 agent
    workflow.add_conditional_edges(
        "tools",
        post_tool_router,
        {
            "human_review": "human_review",
            "agent": "agent",
        },
    )

    # (3) human_review -> agent（回去讓 agent 收尾）
    workflow.add_edge("human_review", "agent")

    app = workflow.compile()
    print(app.get_graph().draw_ascii())
    return app


# =============================
# 7) CLI（stream 展示流程）
# =============================
if __name__ == "__main__":
    print(f"VIP 名單：{VIP_LIST}")
    app = build_app()

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("exit", "q", "quit"):
            break

        # 用 stream 把每個節點輸出顯示出來
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if getattr(msg, "tool_calls", None):
                        print("-> [Agent]: 決定呼叫工具抽取訂單...")
                    else:
                        print(f"-> [Agent]: {msg.content}")

                elif key == "tools":
                    tool_msg = value["messages"][-1]
                    # 工具回傳通常是 ToolMessage，content 會是 JSON 字串
                    print(f"-> [Tools]: {tool_msg.content}")

                elif key == "human_review":
                    print("-> [Human]: 審核完成（已輸入指示）")
