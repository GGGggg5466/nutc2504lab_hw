import json
import os
import re
from typing import TypedDict, Literal, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ====== 不動學長檔案：只 import 使用 ======
from search_searxng import search_searxng  # 學長的 SearXNG 搜尋


# =========================================================
# 0) LLM 設定（用你的 .env）
# =========================================================
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://ws-05.huannago.com/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "Qwen3-VL-8B-Instruct-BF16.gguf")
CACHE_FILE = os.getenv("CACHE_FILE", "verify_cache.json")

# 可選：是否啟用學長的 VLM 讀網頁（預設關閉，避免 api_key="" 問題）
ENABLE_VLM_READ = os.getenv("ENABLE_VLM_READ", "0") == "1"
VLM_TOP_K = int(os.getenv("VLM_TOP_K", "1"))  # 讀前幾個網址（1 就好，避免太慢）

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)


# =========================================================
# 1) State
# =========================================================
class State(TypedDict):
    question: str
    plan: str
    decision: str        # "DIRECT" or "SEARCH"
    query: str
    evidence: str
    answer: str
    is_cache_hit: bool


# =========================================================
# 2) Cache
# =========================================================
def normalize_key(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(key: str, value: dict) -> None:
    data = load_cache()
    data[key] = value
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# 3) Nodes
# =========================================================
def check_cache_node(state: State) -> State:
    key = normalize_key(state["question"])
    cache = load_cache()
    if key in cache:
        hit = cache[key]
        return {
            **state,
            "plan": hit.get("plan", ""),
            "decision": hit.get("decision", ""),
            "query": hit.get("query", ""),
            "evidence": hit.get("evidence", ""),
            "answer": hit.get("answer", ""),
            "is_cache_hit": True,
        }
    return {**state, "is_cache_hit": False}


def planner_node(state: State) -> State:
    """
    產出 decision + plan
    decision = DIRECT / SEARCH
    """
    prompt = f"""你是一個「查證/問答助理」。請先判斷此問題需不需要外部查證搜尋。

問題：{state['question']}

規則：
- 若只是閒聊、或不需要外部事實查證：decision = DIRECT
- 若需要查證（需要外部資料/日期/事實）：decision = SEARCH

請輸出 JSON（只能輸出 JSON，不要多餘文字）：
{{
  "decision": "DIRECT 或 SEARCH",
  "plan": "最多三步的查證/回答計畫"
}}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()

    # 容錯：有些模型可能會多輸出文字，簡單抓 JSON
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        # fallback：當作 SEARCH
        return {**state, "decision": "SEARCH", "plan": text}

    try:
        obj = json.loads(m.group(0))
        decision = str(obj.get("decision", "SEARCH")).upper()
        plan = str(obj.get("plan", "")).strip()
        if decision not in ("DIRECT", "SEARCH"):
            decision = "SEARCH"
        return {**state, "decision": decision, "plan": plan}
    except Exception:
        return {**state, "decision": "SEARCH", "plan": text}


def query_gen_node(state: State) -> State:
    prompt = f"""請根據計畫，生成 1 個最有效的搜尋關鍵字（不要解釋）：

問題：{state['question']}
計畫：{state['plan']}

只輸出關鍵字。"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "query": resp.content.strip()}


def search_tool_node(state: State) -> State:
    """
    不改學長：直接用學長的 search_searxng() 取得結果
    然後整理成 evidence 字串
    （可選）如果 ENABLE_VLM_READ=1，就用學長 vlm_read_website() 讀前幾個網址補摘要
    """
    q = state.get("query", "").strip()
    if not q:
        return {**state, "evidence": "(沒有 query，無法搜尋)"}

    results = search_searxng(q, time_range="day", limit=3)

    if not results:
        return {**state, "evidence": f"(SearXNG 無結果或連線失敗) query={q}"}

    lines = [f"（SearXNG results）query={q}"]
    urls = []

    for i, r in enumerate(results, 1):
        title = (r.get("title") or "（無標題）").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("content") or "").strip().replace("\n", " ")
        snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
        lines.append(f"\n[{i}] {title}\n- url: {url}\n- snippet: {snippet}")
        if url:
            urls.append((title, url))

    # 可選：用學長 VLM 讀網頁（不修改學長檔案）
    if ENABLE_VLM_READ and urls:
        try:
            # 延遲 import：避免沒裝 playwright 就直接炸
            from vlm_read_website import vlm_read_website  # 學長檔案（不改）

            for i, (title, url) in enumerate(urls[:VLM_TOP_K], 1):
                summary = vlm_read_website(url, title=title)
                if summary:
                    lines.append(f"\n[VLM摘要 {i}] {title}\n{summary}")

        except Exception as e:
            lines.append(f"\n⚠️ VLM 讀網頁未啟用或失敗：{e}")

    return {**state, "evidence": "\n".join(lines)}


def final_answer_node(state: State) -> State:
    """
    - Cache hit：answer 已有，直接回
    - DIRECT：直接回答
    - SEARCH：根據 evidence 回答
    最後寫入 cache
    """
    if state.get("answer"):
        return state

    decision = state.get("decision", "SEARCH").upper()

    if decision == "DIRECT":
        prompt = f"""請直接回答（不需要查證）：
問題：{state['question']}
回答要清楚簡潔。"""
    else:
        prompt = f"""你是查證助理，請根據「證據」回答使用者問題。

問題：{state['question']}

證據：
{state.get('evidence', '')}

請輸出：
1) 結論（1句話）
2) 證據摘要（2-3點）
3) 限制/不確定性（若不足要誠實說明）
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    answer = resp.content.strip()

    key = normalize_key(state["question"])
    save_cache(
        key,
        {
            "decision": state.get("decision", ""),
            "plan": state.get("plan", ""),
            "query": state.get("query", ""),
            "evidence": state.get("evidence", ""),
            "answer": answer,
        },
    )

    return {**state, "answer": answer}


# =========================================================
# 4) Routers（照你學長圖：check_cache 分岔 + planner 三岔）
# =========================================================
def cache_router(state: State) -> Literal["planner", "end"]:
    # Cache hit -> END（跟學長一致）
    return "end" if state.get("is_cache_hit", False) else "planner"


def planner_router(state: State) -> Literal["query_gen", "search_tool", "final_answer"]:
    # DIRECT：直接回答
    if state.get("decision", "").upper() == "DIRECT":
        return "final_answer"

    # SEARCH：若還沒 query，先生成 query
    if not state.get("query"):
        return "query_gen"

    # 有 query 但沒有 evidence -> 去 search_tool
    if not state.get("evidence"):
        return "search_tool"

    # 有 evidence -> 收斂回答
    return "final_answer"


# =========================================================
# 5) Build Graph（重點：固定邊只加一次，避免你遇到的重複線）
# =========================================================
workflow = StateGraph(State)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

# check_cache -> planner / END
workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {"planner": "planner", "end": END},
)

# planner 三岔：query_gen / search_tool / final_answer
workflow.add_conditional_edges(
    "planner",
    planner_router,
    {
        "query_gen": "query_gen",
        "search_tool": "search_tool",
        "final_answer": "final_answer",
    },
)

# 固定邊（只加一次！）
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")   # ✅ 搜完回 planner（不接 end、不接 final）
workflow.add_edge("final_answer", END)

app = workflow.compile()


# =========================================================
# 6) Main
# =========================================================
if __name__ == "__main__":
    print(app.get_graph().draw_ascii())
    print(f"\nCACHE_FILE: {os.path.abspath(CACHE_FILE)}")
    print(f"ENABLE_VLM_READ: {ENABLE_VLM_READ}")

    while True:
        q = input("\n輸入問題 (q 離開)：").strip()
        if q.lower() == "q":
            break

        init_state: State = {
            "question": q,
            "plan": "",
            "decision": "",
            "query": "",
            "evidence": "",
            "answer": "",
            "is_cache_hit": False,
        }

        result = app.invoke(init_state)

        print("\n=========== 最終答案 ===========")
        if result.get("is_cache_hit", False):
            # cache hit -> 直接 END，所以這裡從檔案讀 answer 顯示
            key = normalize_key(q)
            hit = load_cache().get(key, {})
            print(hit.get("answer", "(Cache Hit，但快取內沒有 answer?)"))
        else:
            print(result.get("answer", ""))
        print("================================")
        print(f"(Cache Hit: {result.get('is_cache_hit', False)})")
