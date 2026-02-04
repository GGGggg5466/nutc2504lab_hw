import os
import json
from typing import Annotated, TypedDict

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

# ===== Tool（取自 ch4-1）=====
@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """從對話文字提取訂單資訊（姓名、電話、商品、數量、地址）"""
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address,
    }

tools = [extract_order_data]

# ===== LLM =====
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ===== State =====
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===== Node A: agent =====
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ===== Node B: tools =====
tool_node = ToolNode(tools)

# ===== Edge decision =====
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END

# ===== Build graph =====
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

# ✅ 完整版關鍵：工具做完回到 agent 再說人話
workflow.add_edge("tools", "agent")

app = workflow.compile()

def show_event(event: dict):
    """把 stream 的 event 印得像投影片一樣好讀"""
    for node_name, state in event.items():
        print(f"\n--- Node: {node_name} ---")
        last = state["messages"][-1]

        # 如果是 AIMessage 且有 tool_calls，就印 tool_calls
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            print(tool_calls)
        else:
            # ToolMessage / AIMessage 一般回覆
            content = getattr(last, "content", "")
            # 有時 tool message content 是 JSON 字串
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
                if isinstance(parsed, (dict, list)):
                    print(json.dumps(parsed, ensure_ascii=False, indent=2))
                else:
                    print(parsed)
            except Exception:
                print(content)

if __name__ == "__main__":
    print(app.get_graph().draw_ascii())
    print("\n=== ch5-1 full interactive ===")
    print("輸入 exit 或 q 離開。\n")

    # 你可以固定加 system prompt，讓 agent 口氣一致
    sys_msg = SystemMessage(content="你是訂單管理員。若使用者提供訂單資訊就呼叫工具抽取，最後用自然語言回覆確認訂單。")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break

        # stream 會逐節點吐出過程（投影片就是這個）
        for event in app.stream({"messages": [sys_msg, HumanMessage(content=user_input)]}):
            show_event(event)
