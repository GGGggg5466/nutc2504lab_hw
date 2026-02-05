import os
import json
import time
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# =============================
# 0) ENV / Models
# =============================
load_dotenv()

# 慢但強（Expert）：老師常用 ws-02 + gemma
EXPERT_BASE_URL = os.getenv("EXPERT_BASE_URL", os.getenv("BASE_URL", "https://ws-02.wade0426.me/v1"))
EXPERT_API_KEY = os.getenv("EXPERT_API_KEY", os.getenv("API_KEY", ""))
EXPERT_MODEL = os.getenv("EXPERT_MODEL", os.getenv("MODEL", "google/gemma-3-27b-it"))

# 快速通道（Fast）：你自己的 ws-05 + qwen (或任何你可用的快速模型)
FAST_BASE_URL = os.getenv("FAST_BASE_URL", "https://ws-05.huannago.com/v1")
FAST_API_KEY = os.getenv("FAST_API_KEY", os.getenv("API_KEY", ""))
FAST_MODEL = os.getenv("FAST_MODEL", "Qwen3-VL-8B-Instruct-BF16.gguf")

# LLM clients
llm = ChatOpenAI(
    base_url=EXPERT_BASE_URL,
    api_key=EXPERT_API_KEY,
    model=EXPERT_MODEL,
    temperature=0.7,
)

fast_llm = ChatOpenAI(
    base_url=FAST_BASE_URL if FAST_BASE_URL else EXPERT_BASE_URL,
    api_key=FAST_API_KEY if FAST_API_KEY else EXPERT_API_KEY,
    model=FAST_MODEL if FAST_BASE_URL else EXPERT_MODEL,  # 若沒填 fast，就退回 expert
    temperature=0,
)

CACHE_FILE = "qa_cache.json"


# =============================
# 1) Cache helpers
# =============================
def get_clean_key(text: str) -> str:
    # 去空白、去全形標點（簡單版）—跟投影片一致就好
    return (
        text.strip()
        .replace(" ", "")
        .replace("？", "?")
        .replace("！", "!")
        .replace("，", ",")
        .replace("。", ".")
    )


def load_cache() -> dict:
    # 若檔案不存在：建立預設快取（投影片常這樣做示範 Cache Hit）
    if not os.path.exists(CACHE_FILE):
        default_data = {
            get_clean_key("LangGraph是什麼？"): "LangGraph 是一個用「圖（Graph）」來編排 LLM workflow 的框架，支援分支、迴圈、狀態管理與容錯，適合做可控的 agent 系統。",
            get_clean_key("你的名字？"): "我是這個 QA Chat 的 AI 助手～",
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)
        return default_data

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_cache(new_data: dict) -> None:
    current_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except Exception:
            current_data = {}

    current_data.update(new_data)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)


# =============================
# 2) State
# =============================
class State(TypedDict):
    question: str
    answer: str
    source: str  # CACHE / FAST_TRACK_API / LLM_EXPERT


# =============================
# 3) Nodes
# =============================
def check_cache_node(state: State) -> State:
    print(f"\n[系統] 收到問題：{state['question']}")
    cache_data = load_cache()
    clean_query = get_clean_key(state["question"])

    if clean_query in cache_data:
        print("--- 命中快取 (Cache Hit) ---")
        return {
            "answer": cache_data[clean_query],
            "source": "CACHE",
        }

    print("--- 快取未命中 (Cache Miss) ---")
    return {"answer": ""}


def fast_reply_node(state: State) -> State:
    print("--- 進入快速通道 (Fast Track API) ---")
    resp = fast_llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": resp.content.strip(), "source": "FAST_TRACK_API"}


def expert_node(state: State) -> State:
    print("--- 進入專家模式 (LLM Expert) ---")
    prompt = f"請用繁體中文，清楚、有條理地回答下列問題：\n{state['question']}\n"
    # 用 stream 印出（投影片效果）
    chunks = llm.stream([HumanMessage(content=prompt)])

    full_answer = ""
    print("🤖 AI 正在思考並輸出：", end="", flush=True)
    for chunk in chunks:
        content = getattr(chunk, "content", "")
        if content:
            print(content, end="", flush=True)
            full_answer += content
    print("\n")  # 換行

    clean_key = get_clean_key(state["question"])
    save_cache({clean_key: full_answer})
    print(f"--- [系統] 已將完整回答寫入 {CACHE_FILE} ---")

    return {"answer": full_answer, "source": "LLM_EXPERT"}


# =============================
# 4) Router
# =============================
def master_router(state: State) -> Literal["end", "fast", "expert"]:
    # 如果 check_cache 已經有 answer，就結束
    if state.get("answer"):
        return "end"

    q = state["question"]
    # 投影片示範：招呼語走 Fast Track
    if any(word in q for word in ["你好", "嗨", "早安", "哈囉"]):
        return "fast"
    return "expert"


# =============================
# 5) Build Graph
# =============================
def build_app():
    workflow = StateGraph(State)

    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("fast_bot", fast_reply_node)
    workflow.add_node("expert_bot", expert_node)

    workflow.set_entry_point("check_cache")

    workflow.add_conditional_edges(
        "check_cache",
        master_router,
        {
            "end": END,
            "fast": "fast_bot",
            "expert": "expert_bot",
        },
    )

    workflow.add_edge("fast_bot", END)
    workflow.add_edge("expert_bot", END)

    app = workflow.compile()
    print(app.get_graph().draw_ascii())
    return app


# =============================
# 6) CLI
# =============================
if __name__ == "__main__":
    print(f"快取檔案路徑：{os.path.abspath(CACHE_FILE)}")
    print("提示：輸入招呼語（你好/嗨/早安/哈囉）走 Fast Track；一般問題走 Expert；命中 cache 直接回覆。")

    app = build_app()

    while True:
        user_input = input("\n請輸入問題（輸入 q 離開）: ").strip()
        if user_input.lower() == "q":
            break

        inputs: State = {"question": user_input, "answer": "", "source": ""}

        start_time = time.time()
        try:
            result = app.invoke(inputs)
        except Exception as e:
            print(f"發生錯誤：{e}")
            continue
        end_time = time.time()

        print("=" * 30)
        print(f"來源: [{result.get('source', '')}]")
        print(f"耗時: {end_time - start_time:.4f} 秒")

        # Expert 已經在 stream 印過一次，這裡照投影片邏輯可選擇不重印
        if result.get("source") != "LLM_EXPERT":
            print(f"回答:\n{result.get('answer','')}")
        else:
            print("(回答已在上方 streaming 輸出完成)")
