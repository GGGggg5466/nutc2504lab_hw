#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, re
import subprocess
import time
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages


# -----------------------------
# Paths & Env
# -----------------------------
ROOT = Path(__file__).resolve().parent          # meeting_assistant/
TOOLS_DIR = ROOT / "tools"
HW_ASR = TOOLS_DIR / "HW-asr.py"

TOOLS_OUT_DIR = TOOLS_DIR / "out"              # meeting_assistant/tools/out
OUTPUT_DIR = ROOT / "output"                   # meeting_assistant/output

TOOLS_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 只讀 meeting_assistant/.env（不碰外層 Day3_hw/.env）
load_dotenv(ROOT / ".env")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "").strip()

AUDIO_PATH = os.getenv("AUDIO_PATH", str(ROOT / "input" / "Podcast_EP14.wav")).strip()


def _require_env(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Missing env: {name}. Please set it in meeting_assistant/.env")


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _latest_txt(out_dir: Path) -> Path | None:
    txts = list(out_dir.glob("*.txt"))
    if not txts:
        return None
    return max(txts, key=lambda p: p.stat().st_mtime)


def _run_hw_asr() -> Path:
    """
    Run tools/HW-asr.py and return the latest generated .txt path in tools/out/
    """
    if not HW_ASR.exists():
        raise RuntimeError(f"HW ASR script not found: {HW_ASR}")

    src_audio = Path(AUDIO_PATH).expanduser()
    # 支援相對路徑：相對於 meeting_assistant/
    if not src_audio.is_absolute():
        src_audio = (ROOT / src_audio).resolve()

    if not src_audio.exists():
        raise RuntimeError(f"Audio file not found: {src_audio}")
    if src_audio.is_dir():
        raise RuntimeError(f"AUDIO_PATH points to a directory, not a file: {src_audio}")

    TOOLS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(HW_ASR),
        "--audio", str(src_audio),
        "--out-dir", str(TOOLS_OUT_DIR),
        "--save-srt",
    ]
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),            # ✅固定在 meeting_assistant 執行（相對路徑就穩）
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "HW ASR failed.\n"
            f"cmd={cmd}\n"
            f"cwd={ROOT}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )

    latest = _latest_txt(TOOLS_OUT_DIR)
    if not latest or not latest.exists():
        raise RuntimeError(
            "HW ASR succeeded but no .txt found in tools/out.\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )
    return latest


# -----------------------------
# State
# -----------------------------
class MeetingState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    transcript: str
    minutes: str
    summary: str
    report: str


# -----------------------------
# LLM
# -----------------------------
_require_env("OPENAI_BASE_URL", OPENAI_BASE_URL)
_require_env("OPENAI_MODEL", OPENAI_MODEL)

llm = ChatOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,  # proxy 可能允許空值；如用 OpenAI 官方就必須填
    model=OPENAI_MODEL,
    temperature=0,
)

minutes_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一位專業的會議逐字稿整理員。"
         "請根據輸入的逐字稿，輸出「詳細逐字稿（Detailed Minutes）」：\n"
         "1) 以時間軸/段落方式整理（若逐字稿沒有時間戳，就用段落/主題切分）\n"
         "2) 保留關鍵名詞、數字、決策、爭議點\n"
         "3) 請用繁體中文\n"
         "輸出格式用 Markdown。"),
        ("user", "{transcript}"),
    ]
)

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一位會議重點摘要員。"
         "請根據逐字稿輸出「重點摘要（Executive Summary）」：\n"
         "- 3~8 個重點條列\n"
         "- 列出 Action Items（負責人/事項/期限，若未提到就寫未提及）\n"
         "- 請用繁體中文\n"
         "輸出格式用 Markdown。"),
        ("user", "{transcript}"),
    ]
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一位會議文件編輯。"
         "請把「重點摘要」與「詳細逐字稿」整合成一份可交付的會議紀錄文件。\n"
         "要求：\n"
         "- 先放 Executive Summary\n"
         "- 再放 Detailed Minutes\n"
         "- 文件標題：# 會議紀錄\n"
         "- 請用繁體中文、Markdown\n"
         "- 輸出不用太長，請控制在 1200~2000 字左右（重點清楚即可）\n"),
        ("user",
         "## Executive Summary\n{summary}\n\n"
         "## Detailed Minutes\n{minutes}\n"),
    ]
)

minutes_chain = minutes_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()
writer_chain = writer_prompt | llm | StrOutputParser()


# -----------------------------
# Nodes
# -----------------------------
def asr_node(state: MeetingState) -> MeetingState:
    latest_txt = _run_hw_asr()
    transcript = latest_txt.read_text(encoding="utf-8").strip()

    _save_text(OUTPUT_DIR / "transcript.txt", transcript)

    return {
        "transcript": transcript,
        "messages": [HumanMessage(content=f"[ASR 完成] transcript={latest_txt.name}, len={len(transcript)}")],
    }


def minutes_taker_node(state: MeetingState) -> MeetingState:
    transcript = state.get("transcript", "").strip()
    if not transcript:
        raise RuntimeError("minutes_taker_node: transcript is empty")

    minutes = minutes_chain.invoke({"transcript": transcript}).strip()
    _save_text(OUTPUT_DIR / "minutes.md", minutes)

    return {"minutes": minutes, "messages": [HumanMessage(content=f"[Minutes 完成] len={len(minutes)}")]}


def summarizer_node(state: MeetingState) -> MeetingState:
    transcript = state.get("transcript", "").strip()
    if not transcript:
        raise RuntimeError("summarizer_node: transcript is empty")

    summary = summary_chain.invoke({"transcript": transcript}).strip()
    _save_text(OUTPUT_DIR / "summary.md", summary)

    return {"summary": summary, "messages": [HumanMessage(content=f"[Summary 完成] len={len(summary)}")]}


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...內容過長已截斷...]"


def _invoke_with_retry(chain, inp: dict, max_retry: int = 3, sleep_s: int = 3) -> str:
    last_err = None
    for i in range(max_retry):
        try:
            out = chain.invoke(inp)
            return out if isinstance(out, str) else str(out)
        except Exception as e:
            last_err = e
            msg = str(e)
            # Cloudflare 524 / 5xx 常見：回傳一整段 HTML
            if "524" in msg or "cloudflare" in msg.lower() or "5xx" in msg.lower():
                time.sleep(sleep_s * (i + 1))  # 3s, 6s, 9s...
                continue
            raise
    raise RuntimeError(f"writer invoke failed after retries: {last_err}")


def writer_node(state: MeetingState) -> MeetingState:
    minutes = _clip(state.get("minutes", ""), max_chars=8000)
    summary = _clip(state.get("summary", ""), max_chars=4000)

    print("[writer] input chars:", len(minutes), len(summary))

    report = _invoke_with_retry(
        writer_chain,
        {"minutes": minutes, "summary": summary},
        max_retry=3,
        sleep_s=3,
    ).strip()

    _save_text(OUTPUT_DIR / "report.md", report)
    return {"report": report}


# -----------------------------
# Graph
# -----------------------------
def build_app():
    workflow = StateGraph(MeetingState)

    workflow.add_node("asr", asr_node)
    workflow.add_node("minutes_taker", minutes_taker_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("asr")

    workflow.add_edge("asr", "minutes_taker")
    workflow.add_edge("asr", "summarizer")

    workflow.add_edge("minutes_taker", "writer")
    workflow.add_edge("summarizer", "writer")

    workflow.add_edge("writer", END)

    return workflow.compile()

def _print_file(title: str, path: Path, max_chars: int = 12000):
    print("\n" + "=" * 80)
    print(f"[{title}]  {path}")
    print("=" * 80)
    if not path.exists():
        print("(file not found)")
        return
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if len(txt) > max_chars:
        print(txt[:max_chars])
        print(f"\n... (truncated, total_chars={len(txt)})")
    else:
        print(txt)

def _latest_srt(tools_out_dir: Path) -> Path | None:
    if not tools_out_dir.exists():
        return None
    files = sorted(tools_out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

_TIME_RE = re.compile(r"\d\d:\d\d:\d\d[,.]\d+\s+-->\s+\d\d:\d\d:\d\d[,.]\d+")

def _read_text_smart(path: Path) -> str:
    raw = path.read_bytes()
    # BOM detection
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="ignore")
    # heuristic: lots of null bytes => likely utf-16
    if raw.count(b"\x00") > 10:
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")

def _print_srt_pretty(title: str, srt_path: Path, max_lines: int = 3000):
    print("\n" + "=" * 80)
    print(f"{title}: {srt_path}")
    print("=" * 80)

    text = _read_text_smart(srt_path)
    lines = [ln.replace("\ufeff", "").strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln != ""]   # ✅ 去掉所有空白行（間隔就消失）

    out = []
    i = 0
    while i < len(lines):
        # index
        if lines[i].isdigit():
            i += 1
        if i >= len(lines):
            break

        # time
        if not _TIME_RE.search(lines[i]):
            i += 1
            continue
        t = lines[i].replace(",", ".")
        t = t.replace(" --> ", "-")
        i += 1

        # text lines: 直到下一個 time 或下一個 index
        texts = []
        while i < len(lines) and (not _TIME_RE.search(lines[i])) and (not lines[i].isdigit()):
            texts.append(lines[i])
            i += 1

        if texts:
            out.append(f"{t} {' '.join(texts)}")
            if len(out) >= max_lines:
                out.append(f"... (truncated, shown_lines={len(out)})")
                break

    print("\n".join(out))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app = build_app()

    print(app.get_graph().draw_ascii())
    print("\n=== meeting assistant run ===\n")

    init_state: MeetingState = {"messages": [HumanMessage(content="start")]}
    final = app.invoke(init_state)

    # ===== DEBUG：確認你正在讀哪個 tools/out =====
    print("\n[DEBUG] BASE_DIR      =", ROOT)
    print("[DEBUG] TOOLS_OUT_DIR =", TOOLS_OUT_DIR)
    print("[DEBUG] OUTPUT_DIR    =", OUTPUT_DIR)

    # 1) 重點摘要（summary）
    _print_file("Executive Summary (from output/summary.md)", OUTPUT_DIR / "summary.md")

    # 2) 逐字稿（latest srt）
    latest = _latest_srt(TOOLS_OUT_DIR)
    print(f"[DEBUG] latest srt = {latest}")  # 你現在已有這行，很好
    if latest:
        _print_srt_pretty("Detailed Transcript (latest .srt)", latest, max_lines=120)
    else:
        print("(no .srt found in tools/out)")