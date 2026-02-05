import os
import json
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# =============================
# 0) LLM / Env
# =============================
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://ws-02.wade0426.me/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "google/gemma-3-27b-it")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0.7,  # 投影片 7_1 通常會調高一點
)

CACHE_FILE = "translation_cache.json"


# =============================
# 1) Cache Utils
# =============================
def load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(original: str, translated: str) -> None:
    data = load_cache()
    data[original] = translated
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =============================
# 2) State
# =============================
class State(TypedDict):
    original_text: str
    translated_text: str
    critique: str
    attempts: int
    is_cache_hit: bool  # 命中快取？


# =============================
# 3) Nodes
# =============================
def check_cache_node(state: State) -> State:
    print("\n--- 檢查快取 (Check Cache) ---")
    data = load_cache()
    original = state["original_text"]

    if original in data:
        print("✅ 命中快取！直接回傳結果")
        return {
            "translated_text": data[original],
            "is_cache_hit": True,
        }

    print("❌ 未命中快取，準備開始翻譯流程...")
    return {"is_cache_hit": False}


def translator_node(state: State) -> State:
    print(f"\n--- 翻譯嘗試（第 {state['attempts'] + 1} 次）---")

    prompt = (
        "你是一名翻譯員，請將以下中文翻譯成英文，"
        "不要任何解釋，只輸出英文翻譯：\n"
        f"{state['original_text']}"
    )

    if state["critique"].strip():
        prompt += (
            f"\n\n上一次審查意見：{state['critique']}\n"
            "請根據審查意見修正翻譯。"
        )

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "translated_text": resp.content.strip(),
        "attempts": state["attempts"] + 1,
    }


def reflector_node(state: State) -> State:
    print("\n--- 審查中 (Reflection) ---")
    print(f"翻譯：{state['translated_text']}")

    prompt = f"""
你是一個嚴格的翻譯審查員。
原文：{state['original_text']}
譯文：{state['translated_text']}

請檢查譯文是否準確且自然。
規則：
- 如果譯文已經很完美，請只回覆：PASS
- 否則請給出簡短、具體、可操作的修改建議
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": resp.content.strip()}


# =============================
# 4) Routers
# =============================
def cache_router(state: State) -> Literal["end", "translator"]:
    # 命中快取：直接結束；未命中：進翻譯
    if state["is_cache_hit"]:
        return "end"
    return "translator"


def critique_router(state: State) -> Literal["translator", "end"]:
    critique = state["critique"].strip().upper()

    if "PASS" in critique:
        print("\n--- 審查通過！---")
        return "end"

    if state["attempts"] >= 3:
        print("\n--- 達到最大重試次數，強制結束 ---")
        return "end"

    print(f"\n--- 審查未通過：{state['critique']} ---")
    print("--- 退回重翻 ---")
    return "translator"


# =============================
# 5) Build Graph
# =============================
def build_app():
    workflow = StateGraph(State)

    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("reflector", reflector_node)

    workflow.set_entry_point("check_cache")

    # check_cache -> (hit end / miss translator)
    workflow.add_conditional_edges(
        "check_cache",
        cache_router,
        {"end": END, "translator": "translator"},
    )

    # normal path: translator -> reflector -> (translator or end)
    workflow.add_edge("translator", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        critique_router,
        {"translator": "translator", "end": END},
    )

    app = workflow.compile()
    print(app.get_graph().draw_ascii())
    return app


# =============================
# 6) CLI
# =============================
if __name__ == "__main__":
    print(f"快取檔案：{CACHE_FILE}")
    app = build_app()

    while True:
        user_input = input("\n請輸入要翻譯的中文（exit/q 離開）: ").strip()
        if user_input.lower() in ("exit", "q", "quit"):
            break

        inputs: State = {
            "original_text": user_input,
            "translated_text": "",
            "critique": "",
            "attempts": 0,
            "is_cache_hit": False,
        }

        result = app.invoke(inputs)

        # cache miss 才寫入（命中就不用寫）
        if not result.get("is_cache_hit", False):
            save_cache(result["original_text"], result["translated_text"])
            print("（已將翻譯結果寫入快取）")

        print("\n========== 最終結果 ==========")
        print("原文：", result["original_text"])
        print("譯文：", result["translated_text"])
        print("來源：", "快取(Cache)" if result.get("is_cache_hit") else "生成(LLM)")
        print("嘗試次數：", result.get("attempts", 0))
        print("審查結果：", result.get("critique", ""))
