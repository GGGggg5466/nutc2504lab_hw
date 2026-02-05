import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# =============================
# Env / LLM
# =============================
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

# =============================
# 1) State
# =============================
class State(TypedDict):
    original_text: str
    translated_text: str
    critique: str
    attempts: int  # 重試次數（防止無限迴圈）

# =============================
# 2) Nodes
# =============================
def translator_node(state: State) -> State:
    """負責翻譯的節點"""
    print(f"\n--- 翻譯嘗試（第 {state['attempts'] + 1} 次）---")

    prompt = (
        "你是一名翻譯員，請將以下中文翻譯成英文，"
        "不要任何解釋，只輸出英文翻譯：\n"
        f"{state['original_text']}"
    )

    # 若有審查建議，要求修正
    if state["critique"].strip():
        prompt += f"\n\n上一次的審查意見是：{state['critique']}\n請根據審查意見修正翻譯。"

    resp = llm.invoke([HumanMessage(content=prompt)])

    return {
        "translated_text": resp.content.strip(),
        "attempts": state["attempts"] + 1,
    }

def reflector_node(state: State) -> State:
    """負責審查的節點（Critique）"""
    print("\n--- 審查中（Reflection）---")
    print(f"翻譯：{state['translated_text']}")

    prompt = f"""
你是一個嚴格的翻譯審查員。
原文：{state['original_text']}
譯文：{state['translated_text']}

請檢查譯文是否：
1) 忠實傳達原意
2) 語氣自然
3) 沒有誤譯（尤其是反諷、語氣、暗示）

規則：
- 如果譯文已經很完美，請只回覆：PASS
- 如果需要修改，請給出簡短、具體、可操作的修改建議（不要長篇大論）

如果譯文中出現 "perfect time" 且原文語氣為抱怨或反諷，
即使語法正確，也必須判定為未通過。
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": resp.content.strip()}

# =============================
# 3) Edge (Router)
# =============================
def should_continue(state: State) -> Literal["translator", "end"]:
    critique = state["critique"].strip().upper()

    if "PASS" in critique:
        print("\n--- 審查通過！---")
        return "end"

    if state["attempts"] >= 3:
        print("\n--- 達到最大重試次數，強制結束 ---")
        return "end"

    print(f"\n--- 審查未通過：{state['critique']} ---")
    print("--- 退回重寫 ---")
    return "translator"

# =============================
# 4) Build Graph
# =============================
def build_app():
    workflow = StateGraph(State)

    workflow.add_node("translator", translator_node)
    workflow.add_node("reflector", reflector_node)

    workflow.set_entry_point("translator")

    workflow.add_edge("translator", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {"translator": "translator", "end": END},
    )

    app = workflow.compile()
    print(app.get_graph().draw_ascii())
    return app

# =============================
# 5) CLI
# =============================
if __name__ == "__main__":
    app = build_app()

    while True:
        user_input = input("\nUser（輸入中文，exit/q 離開）: ").strip()
        if user_input.lower() in ("exit", "q", "quit"):
            break

        inputs: State = {
            "original_text": user_input,
            "translated_text": "",
            "critique": "",
            "attempts": 0,
        }

        result = app.invoke(inputs)

        print("\n========== 最終結果 ==========")
        print("原文：", result["original_text"])
        print("最終譯文：", result["translated_text"])
        print("最終次數：", result["attempts"])
        print("最終審查：", result["critique"])
