import os
import json
import time
from typing import TypedDict, List, Dict, Any, Literal, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# =========================
# 0) 引用學長的檔案（不修改）
# =========================
# 你的資料夾若是：
# Day4_hw/
#   homework.py   (本檔)
#   search_searxng.py
#   vlm_read_website.py
#
# 直接 import 就會抓到同資料夾的模組
from search_searxng import search_searxng
from vlm_read_website import vlm_read_website


# =========================
# 1) 環境變數 / 設定
# =========================
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://ws-05.huannago.com/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "Qwen3-VL-8B-Instruct-BF16.gguf")

CACHE_FILE = os.getenv("CACHE_FILE", "verify_cache.json")
ENABLE_VLM_READ = os.getenv("ENABLE_VLM_READ", "False").lower() == "true"

print(f"CACHE_FILE: {os.path.abspath(CACHE_FILE)}")
print(f"ENABLE_VLM_READ: {ENABLE_VLM_READ}")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0.2,
)


# =========================
# 2) Cache utilities
# =========================
def get_clean_key(text: str) -> str:
    """讓 cache key 更穩：去頭尾空白、統一全形/半形空白、移除多餘換行"""
    return " ".join(text.strip().split())

def load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(key: str, value: Dict[str, Any]) -> None:
    data = load_cache()
    data[key] = value
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 3) State 定義
# =========================
class State(TypedDict):
    question: str
    answer: str
    cache_hit: bool

    # knowledge_base：累積證據/資料（可被 planner 判斷是否足夠）
    kb: List[Dict[str, Any]]

    # search related
    search_query: str
    loop: int

    # 最終輸出附帶資訊
    evidence_summary: str


# =========================
# 4) Nodes
# =========================
def check_cache_node(state: State) -> State:
    q = get_clean_key(state["question"])
    cache = load_cache()

    print("\n--- [check_cache] 檢查快取 ---")
    if q in cache:
        print("✅ Cache Hit")
        state["cache_hit"] = True
        state["answer"] = cache[q].get("answer", "")
        state["evidence_summary"] = cache[q].get("evidence_summary", "")
        # kb 也可以帶回來（可選）
        state["kb"] = cache[q].get("kb", [])
    else:
        print("❌ Cache Miss")
        state["cache_hit"] = False
    return state


def planner_node(state: State) -> State:
    """
    核心：用 knowledge_base 判斷「夠不夠回答」
    - 夠：走 final_answer
    - 不夠：走 query_gen（產生關鍵字再去 search_tool）
    """
    print("\n--- [planner] 評估 knowledge_base 是否足夠 ---")

    q = state["question"]
    kb = state.get("kb", [])
    loop = state.get("loop", 0)

    # safety：避免無限循環
    if loop >= 3:
        print("⚠️ loop >= 3，強制進 final_answer（用現有資料整理出可回答的版本）")
        return state

    # 如果 kb 空，直接判定不夠 -> 去 query_gen
    if not kb:
        print("kb 目前是空的 → 不足，準備去 query_gen")
        return state

    # 用 LLM 判斷「資料是否足夠」
    kb_text = "\n".join(
        [f"- title: {x.get('title','')}\n  url: {x.get('url','')}\n  snippet: {x.get('snippet','')}"
         for x in kb[:8]]
    )

    prompt = f"""
你是一個嚴謹的查證助理，請判斷目前 knowledge_base 是否足夠回答使用者問題。

【問題】
{q}

【knowledge_base 摘要】
{kb_text}

請只輸出 JSON（不要多餘文字）：
{{
  "enough": true/false,
  "why": "一句話理由",
  "next": "FINAL" 或 "SEARCH"
}}
"""
    resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # 解析 JSON（容錯）
    enough = False
    try:
        j = json.loads(resp)
        enough = bool(j.get("enough", False))
        why = j.get("why", "")
        nxt = j.get("next", "SEARCH")
    except Exception:
        why = "LLM 回傳非 JSON，保守視為不足"
        nxt = "SEARCH"

    print(f"planner 判斷：enough={enough} / next={nxt} / why={why}")

    return state


def query_gen_node(state: State) -> State:
    """
    生成關鍵字（或搜尋 query），交給 search_tool 使用。
    """
    print("\n--- [query_gen] 生成搜尋關鍵字 ---")

    q = state["question"]
    prompt = f"""
你是搜尋關鍵字生成器。請針對問題產生 1 條最適合拿去 searXNG 搜尋的查證關鍵字/查詢句。
要求：
- 不要太長（20~60字）
- 優先使用可查證的官方關鍵字（機關/統計/公告/報告）
- 只輸出一行 query（不要解釋）

問題：{q}
"""
    search_query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    state["search_query"] = search_query
    print(f"search_query = {search_query}")
    return state


def search_tool_node(state: State) -> State:
    """
    用學長的 search_searxng() 去抓資料，寫入 knowledge_base。
    注意：此 node 不會直接走 final/end，會回到 planner 再評估。
    """
    print("\n--- [search_tool] searXNG 搜尋 ---")

    query = state.get("search_query", "").strip()
    if not query:
        # 沒 query 就回去（讓 planner 再決策或最後兜底）
        print("⚠️ search_query 空的，略過搜尋")
        return state

    results = search_searxng(query=query, limit=5)

    kb = state.get("kb", [])
    added = 0
    for r in results or []:
        item = {
            "title": r.get("title", ""),
            "url": r.get("url", r.get("link", "")),
            "snippet": r.get("snippet", r.get("content", "")),
            "query": query,
            "ts": time.time(),
        }
        kb.append(item)
        added += 1

    state["kb"] = kb
    state["loop"] = state.get("loop", 0) + 1
    print(f"已加入 kb：{added} 筆（loop={state['loop']}）")

    return state


def final_answer_node(state: State) -> State:
    """
    整理最終答案：
    - 若 cache_hit=True：answer 已有（可選擇再包裝）
    - 否則用 kb 產生「結論 / 證據摘要 / 限制」
    並寫入 cache。
    """
    print("\n--- [final_answer] 輸出最終答案 ---")

    q = state["question"]
    kb = state.get("kb", [])

    # 若是 Cache Hit 直接回傳（仍可補一行資訊）
    if state.get("cache_hit"):
        state["answer"] = state.get("answer", "")
        state["evidence_summary"] = state.get("evidence_summary", "")
        return state

    kb_text = "\n".join(
        [f"[{i+1}] {x.get('title','')}\nURL: {x.get('url','')}\n摘要: {x.get('snippet','')}"
         for i, x in enumerate(kb[:8])]
    )

    # 可選：若問題帶 URL 且 ENABLE_VLM_READ=True，可讀網頁補 kb（這段不影響主流程）
    if ENABLE_VLM_READ:
        # 粗略抓 URL
        import re
        m = re.search(r"https?://\S+", q)
        if m:
            url = m.group(0)
            print(f"🔎 VLM 讀網頁：{url}")
            try:
                text = vlm_read_website(url)
                state["kb"].append({"title": "VLM 網頁摘錄", "url": url, "snippet": text[:800]})
            except Exception as e:
                print(f"VLM 讀取失敗：{e}")

    prompt = f"""
你是一個「自動查證 AI」。請用目前資料回答問題，格式固定如下：

1) 結論：一句話回答（避免誇大，若不確定要說不確定）
2) 證據摘要：列出 2~4 點，來源只用 knowledge_base（可用「從某網站/某單位」描述，不必真的引用格式）
3) 限制/不確定性：說明資料缺口或可能誤差，給下一步建議

【問題】
{q}

【knowledge_base】
{kb_text}
"""
    ans = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    state["answer"] = ans

    # 證據摘要另外存一份（給 README 或 debug 用）
    ev_prompt = f"""
請把 knowledge_base 內容濃縮成 3~6 行「證據摘要」，每行包含：
- 來源/網站名（或標題）
- 大致說明它支持了什麼
只輸出摘要（不要多餘文字）。

knowledge_base：
{kb_text}
"""
    evidence_summary = llm.invoke([HumanMessage(content=ev_prompt)]).content.strip()
    state["evidence_summary"] = evidence_summary

    # 寫入 cache
    key = get_clean_key(q)
    save_cache(key, {
        "question": q,
        "answer": ans,
        "evidence_summary": evidence_summary,
        "kb": kb,
        "ts": time.time()
    })
    print("✅ 已寫入快取")

    return state


# =========================
# 5) Routers（決策邏輯）
# =========================
def after_cache_router(state: State) -> Literal["final_answer", "planner"]:
    # hit -> 直接 final_answer（裡面會直接回傳 cache 的答案）
    if state.get("cache_hit"):
        return "final_answer"
    return "planner"


def master_router(state: State) -> Literal["final_answer", "query_gen"]:
    """
    planner 之後的路由（照你要的 3 路邏輯）：
    - 若 kb 足夠 → final_answer
    - 否則 → query_gen（再去 search_tool）
    """
    # loop 防呆：planner_node 已經有保護，這裡補一層
    if state.get("loop", 0) >= 3:
        return "final_answer"

    # kb 空就必須 query_gen
    if not state.get("kb"):
        return "query_gen"

    # 再做一次輕量判斷：讓 planner_node 已經判斷過的結果生效
    # 這裡最保守：只要 kb 有資料，但仍不足，就 query_gen
    # 我們用同一個判斷 prompt 再跑一次會浪費 token，所以改用簡化規則：
    # - kb >= 3 筆：先嘗試 final_answer（多數可回答）
    # - 否則：先 query_gen 再補資料
    if len(state["kb"]) >= 3:
        return "final_answer"
    return "query_gen"


# =========================
# 6) 組 Graph（保證不會多接 search->final/end）
# =========================
workflow = StateGraph(State)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

# check_cache -> (hit) final_answer -> END
# check_cache -> (miss) planner
workflow.add_conditional_edges(
    "check_cache",
    after_cache_router,
    {
        "final_answer": "final_answer",
        "planner": "planner",
    }
)

# planner -> (enough) final_answer
# planner -> (not enough) query_gen
workflow.add_conditional_edges(
    "planner",
    master_router,
    {
        "final_answer": "final_answer",
        "query_gen": "query_gen",
    }
)

# query_gen -> search_tool -> planner（只回 planner）
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")

# final_answer -> END
workflow.add_edge("final_answer", END)

app = workflow.compile()


# =========================
# 7) Debug / 檢查工具
# =========================
def print_edges():
    g = app.get_graph()
    print("\nEdges:")
    for e in sorted(g.edges, key=lambda x: (x.source, x.target)):
        print(" ", e)

def print_mermaid():
    # 這會輸出 Mermaid 原始碼（沒有圖是正常的，因為 terminal 不會渲染）
    try:
        g = app.get_graph()
        print(g.draw_mermaid())
    except Exception as e:
        print("draw_mermaid 失敗：", e)


# =========================
# 8) CLI
# =========================
if __name__ == "__main__":
    # 你要確認線路對不對 → 先印 edges 最準
    print_edges()
    print(app.get_graph().draw_ascii())
    # 想要 Mermaid 原始碼也可以印（貼到支援 Mermaid 的地方才會變圖）
    # print_mermaid()

    while True:
        user_input = input("\n輸入問題（q 離開）：").strip()
        if user_input.lower() in ["q", "quit", "exit"]:
            break

        inputs: State = {
            "question": user_input,
            "answer": "",
            "cache_hit": False,
            "kb": [],
            "search_query": "",
            "loop": 0,
            "evidence_summary": "",
        }

        start = time.time()
        result = app.invoke(inputs)
        end = time.time()

        print("\n========== 最終答案 ==========")
        print(result["answer"])
        print("\n( Cache Hit:", result.get("cache_hit", False), ")")
        print("( 耗時: %.3f 秒 )" % (end - start))
