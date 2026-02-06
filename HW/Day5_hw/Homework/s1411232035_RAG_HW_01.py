#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1411232035_RAG_HW_01.py (Qdrant retrieval version)

- Read questions.csv (q_id, questions, answer, source)
- For each method (fixed/sliding/semantic):
    - Embed the question via EMBED_URL
    - Search top-1 from corresponding Qdrant collection
    - Submit retrieve_text to scoring API
    - Write 60 rows CSV (utf-8-sig)

Prereqs:
    pip install qdrant-client requests pandas

Qdrant collections should already be indexed:
    day5_fixed / day5_sliding / day5_semantic
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from qdrant_client import QdrantClient


# ====== APIs ======
SCORE_URL_DEFAULT = "https://hw-01.wade0426.me/submit_answer"
EMBED_URL_DEFAULT = "https://ws-04.wade0426.me/embed"
TASK_DESC_DEFAULT = "檢索技術文件"

QDRANT_URL_DEFAULT = "http://localhost:6333"

# ====== Methods / Collections ======
METHODS: List[Tuple[str, str]] = [
    ("固定大小", "day5_fixed"),
    ("滑動視窗", "day5_sliding"),
    ("語意切塊", "day5_semantic"),
]


@dataclass
class QuestionItem:
    q_id: int
    question: str


def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def load_questions_csv(path: str) -> List[QuestionItem]:
    # 作業常見 utf-8-sig（有 BOM），用這個最保險
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 你的欄位目前是：q_id, questions, answer, source
    # 兼容少數同學的欄位：id -> q_id
    if "q_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "q_id"})
    if "questions" not in df.columns:
        raise KeyError(f"questions.csv 找不到欄位 'questions'，目前欄位={df.columns.tolist()}")

    items: List[QuestionItem] = []
    for _, row in df.iterrows():
        qid_raw = row.get("q_id", "")
        qid = int(qid_raw)
        qtext = _safe_str(row.get("questions", "")).strip()
        if qtext:
            items.append(QuestionItem(q_id=qid, question=qtext))
    return items


def make_requests_session() -> requests.Session:
    s = requests.Session()
    # 可視需要加 headers（目前 API 不要求）
    return s


def embed_one(session: requests.Session, embed_url: str, task_desc: str, text: str, timeout: int = 60) -> List[float]:
    payload = {"texts": [text], "task_description": task_desc, "normalize": True}
    r = session.post(embed_url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "embeddings" not in data:
        raise ValueError(f"embed API 回傳沒有 embeddings：{data}")
    return data["embeddings"][0]


def score_api(session: requests.Session, score_url: str, q_id: int, student_answer: str, timeout: int = 60) -> float:
    payload = {"q_id": int(q_id), "student_answer": student_answer}
    r = session.post(score_url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # 常見：{"score": 0.87} 或 {"data":{"score":...}}
    if "score" in data:
        return float(data["score"])
    if "data" in data and isinstance(data["data"], dict) and "score" in data["data"]:
        return float(data["data"]["score"])

    raise ValueError(f"score API 回傳格式看不到 score：{data}")


def qdrant_search_top1(client, collection: str, query_vector):
    # 新版：client.search(...)
    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=1,
            with_payload=True,
        )
        if not hits:
            return "", ""
        pay = hits[0].payload or {}
        return pay.get("text", ""), pay.get("source", "")

    # 舊版：client.search_points(...)
    if hasattr(client, "search_points"):
        res = client.search_points(
            collection_name=collection,
            vector=query_vector,
            limit=1,
            with_payload=True,
        )
        # 有些版本回傳物件帶 points，有些直接 list
        hits = res.points if hasattr(res, "points") else res
        if not hits:
            return "", ""
        pay = hits[0].payload or {}
        return pay.get("text", ""), pay.get("source", "")

    # 部分版本：client.query_points(...)
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=1,
            with_payload=True,
        )
        hits = res.points if hasattr(res, "points") else []
        if not hits:
            return "", ""
        pay = hits[0].payload or {}
        return pay.get("text", ""), pay.get("source", "")

    raise RuntimeError("你的 qdrant-client 版本沒有 search/search_points/query_points，請升級 qdrant-client。")



def run(
    questions_csv: str,
    out_csv: str,
    qdrant_url: str,
    embed_url: str,
    score_url: str,
    task_desc: str,
    top_k: int = 1,  # 目前作業只需要 top-1
    sleep_sec: float = 0.0,  # 避免 API 被打太快
) -> None:
    if top_k != 1:
        raise ValueError("此作業流程預設 top_k=1（只取 top-1 chunk）。")

    questions = load_questions_csv(questions_csv)
    print(f"[INFO] loaded questions: {len(questions)} from {questions_csv}")

    session = make_requests_session()
    qdrant = QdrantClient(url=qdrant_url)

    # 小優化：同一題的 embedding 三個 method 共用（省 2 次 embed 呼叫）
    qid_to_vec: Dict[int, List[float]] = {}

    out_rows: List[Dict[str, object]] = []
    for qi in questions:
        if qi.q_id not in qid_to_vec:
            qid_to_vec[qi.q_id] = embed_one(session, embed_url, task_desc, qi.question)
            if sleep_sec:
                time.sleep(sleep_sec)

        q_vec = qid_to_vec[qi.q_id]

        for method_name, collection in METHODS:
            retrieve_text, source = qdrant_search_top1(qdrant, collection, q_vec)

            # 如果真的搜不到（理論上不會），給一個最小字串避免 API 失敗
            if not retrieve_text.strip():
                retrieve_text = "（空）"

            score = score_api(session, score_url, qi.q_id, retrieve_text)
            if sleep_sec:
                time.sleep(sleep_sec)

            out_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "q_id": qi.q_id,
                    "method": method_name,
                    "retrieve_text": retrieve_text,
                    "score": score,
                    "source": source,
                }
            )

    # 輸出 utf-8-sig（Excel 不亂碼）
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "q_id", "method", "retrieve_text", "score", "source"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] rows={len(out_rows)} -> {out_csv}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Day5 RAG HW01 - Qdrant retrieval version")
    ap.add_argument("--questions", default="questions.csv", help="questions.csv path")
    ap.add_argument("--out", default="s1411232035_RAG_HW_01_qdrant.csv", help="output csv path")
    ap.add_argument("--qdrant-url", default=QDRANT_URL_DEFAULT, help="Qdrant URL (e.g., http://localhost:6333)")
    ap.add_argument("--embed-url", default=EMBED_URL_DEFAULT, help="Embedding API URL")
    ap.add_argument("--score-url", default=SCORE_URL_DEFAULT, help="Scoring API URL")
    ap.add_argument("--task-desc", default=TASK_DESC_DEFAULT, help="Embedding task_description")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between API calls (avoid rate limit)")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    run(
        questions_csv=args.questions,
        out_csv=args.out,
        qdrant_url=args.qdrant_url,
        embed_url=args.embed_url,
        score_url=args.score_url,
        task_desc=args.task_desc,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    main()
