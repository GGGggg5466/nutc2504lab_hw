import time
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


# ===== 你的 vLLM OpenAI-compatible endpoint =====
BASE_URL = "https://ws-02.wade0426.me/v1"
API_KEY = "vllm-token"
MODEL = "google/gemma-3-27b-it"   # 先用你目前跑得動的；之後再換更大的


def build_chain() -> RunnableParallel:
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=0.0,   # 投影片要求：固定輸出、好觀察
    )

    parser = StrOutputParser()

    ig_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是IG小編。請用活潑口吻、帶emoji、加上3-6個hashtag。"
         "字數約60-120字，使用繁體中文。"),
        ("human", "主題：{topic}")
    ])

    li_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是LinkedIn專業小編。口吻專業、結構清楚，分2-4點條列。"
         "字數約120-200字，使用繁體中文，不要emoji。"),
        ("human", "主題：{topic}")
    ])

    ig_chain = ig_prompt | llm | parser
    li_chain = li_prompt | llm | parser

    # RunnableParallel：同一個輸入 topic，同時跑兩條 chain
    parallel = RunnableParallel(instagram=ig_chain, linkedin=li_chain)
    return parallel


def stream_mode(parallel: RunnableParallel, topic: str) -> None:
    """
    Streaming（交錯輸出）：instagram / linkedin 會交錯吐出
    但我們做「行緩衝」：累積到換行或句尾符號才印，畫面會直向可讀。
    """
    print("\n=== Streaming（不同主題交錯輸出）===\n")

    buffers = {"instagram": "", "linkedin": ""}

    # 你可以調整這個：越小越常換行（越碎），越大越像整段再吐
    MIN_CHARS_TO_FLUSH = 25

    # 觸發 flush 的條件：換行 或 句尾/分隔符號
    END_TRIGGERS = ("\n", "。", "！", "？", "；", "：", "…")

    def flush(k: str, force: bool = False):
        s = buffers[k]
        if not s.strip():
            buffers[k] = ""
            return
        if force or len(s) >= MIN_CHARS_TO_FLUSH or any(t in s for t in END_TRIGGERS):
            # 以「一行」輸出，避免 end="" 造成橫向黏貼
            # 把多餘換行壓成空白，輸出更乾淨
            line = " ".join(s.splitlines()).strip()
            if line:
                print(f"[{k}] {line}", flush=True)
            buffers[k] = ""

    for event in parallel.stream({"topic": topic}):
        for k, v in event.items():
            # v 是增量 token/片段
            buffers[k] += v

            # 若這次增量本身包含換行，優先 flush
            if "\n" in v:
                flush(k, force=True)
            else:
                # 否則看 buffer 長度或句尾符號
                flush(k, force=False)

    # 結束時把剩下的都印出來
    flush("instagram", force=True)
    flush("linkedin", force=True)

    print("\n=== Streaming Done ===\n")




def batch_mode(parallel: RunnableParallel, topic: str) -> None:
    """
    Batch：一次拿到完整結果 + 計時
    """
    print("\n=== Batch（一次輸出 + 計時）===\n")
    t0 = time.perf_counter()
    out: Dict[str, str] = parallel.invoke({"topic": topic})
    dt = time.perf_counter() - t0

    print(f"耗時：{dt:.2f} 秒\n")
    print("【LinkedIn 專業說】")
    print(out["linkedin"].strip())
    print("\n" + "-" * 60 + "\n")
    print("【IG 網紅說】")
    print(out["instagram"].strip())
    print("\n=== Batch Done ===\n")


def main():
    parallel = build_chain()

    # CLI 互動
    while True:
        topic = input("輸入主題 Topic（exit/q 離開）：").strip()
        if topic.lower() in ("exit", "q"):
            print("Bye!")
            return

        # 1) Streaming 交錯輸出
        stream_mode(parallel, topic)

        # 2) Batch 一次輸出（含計時）
        batch_mode(parallel, topic)


if __name__ == "__main__":
    main()
