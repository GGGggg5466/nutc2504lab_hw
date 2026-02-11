#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# =========================
# Embedding + LLM client
# =========================

def embed_query(embed_url: str, text: str, task_description: str = "檢索技術文件", normalize: bool = True, timeout: int = 120) -> List[float]:
    resp = requests.post(
        embed_url,
        json={"texts": [text], "task_description": task_description, "normalize": normalize},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"][0]
    if isinstance(data, list):
        return data[0]
    raise ValueError("Unexpected embed response format")

def llm_chat(llm_url: str, model: str, api_key: str, messages: List[Dict[str, str]], temperature: float = 0.1, timeout: int = 180) -> str:
    """
    OpenAI-compatible chat completions:
      POST llm_url
      headers: Authorization: Bearer <key>
      json: {model, messages, temperature}
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # 學長新增的推理控制參數
        "detailed_thinking": "off",
    }
    resp = requests.post(llm_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Common formats
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        return (msg.get("content") or "").strip()

    raise ValueError(f"Unexpected LLM response: keys={list(data.keys()) if isinstance(data, dict) else type(data)}")


# =========================
# Retrieval
# =========================

def retrieve_context(
    client: QdrantClient,
    collection: str,
    query_vec: List[float],
    top_k: int = 10,
    exclude_injection: bool = True,
) -> List[Dict[str, Any]]:
    flt = None
    if exclude_injection:
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="is_injection",
                    match=qmodels.MatchValue(value=False),
                )
            ]
        )

    # ---- Qdrant client API compatibility layer ----
    hits = []

    # Preferred (qdrant-client 1.10+): query_points
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )
        hits = getattr(res, "points", []) or []

    # Alternative: search_points
    elif hasattr(client, "search_points"):
        res = client.search_points(
            collection_name=collection,
            vector=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )
        # Some versions return .result, others return list directly
        hits = getattr(res, "result", res)
        if hits is None:
            hits = []
        # If it's not a list, try points
        if not isinstance(hits, list):
            hits = getattr(res, "points", []) or []

    # Legacy: search (your env doesn't have it, but keep for completeness)
    elif hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        ) or []

    else:
        raise AttributeError(
            "QdrantClient has no query_points/search_points/search. "
            "Please upgrade qdrant-client or adjust retrieval API."
        )

    contexts: List[Dict[str, Any]] = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        score = getattr(h, "score", None)
        contexts.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "page": payload.get("page", None),
                "score": float(score) if score is not None else 0.0,
            }
        )
    return contexts


def format_context_for_prompt(ctxs: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(ctxs, 1):
        src = c.get("source", "")
        page = c.get("page", None)
        tag = f"{src}" + (f" p.{page}" if page else "")
        text = (c.get("text", "") or "").strip()
        lines.append(f"[{i}] ({tag}) {text}")
    return "\n\n".join(lines).strip()

def sources_compact(ctxs: List[Dict[str, Any]]) -> str:
    uniq = []
    seen = set()
    for c in ctxs:
        src = c.get("source", "")
        page = c.get("page", None)
        s = f"{src}" + (f":{page}" if page else "")
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return "; ".join(uniq)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/test_dataset.csv", help="CSV with q_id + questions OR questions column")
    ap.add_argument("--output", default="outputs/test_dataset.csv", help="Output CSV (q_id,questions,answer,source)")
    ap.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "day7_collection_v2"))
    ap.add_argument("--embed_url", default=os.getenv("EMBED_URL", "https://ws-04.wade0426.me/embed"))
    ap.add_argument("--llm_url", default=os.getenv("LLM_URL", "https://ws-05.huannago.com/v1/chat/completions"))
    ap.add_argument("--llm_model", default=os.getenv("LLM_MODEL", "Qwen3-VL-8B-Instruct-BF16.gguf"))
    ap.add_argument("--llm_key", default=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    ap.add_argument("--top_k", type=int, default=int(os.getenv("TOP_K", "10")))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.1")))
    ap.add_argument("--pred_qa_output", default="outputs/pred_questions_answer.csv", help="Extra file for deepeval (q_id,questions,answer)")
    args = ap.parse_args()

    if not args.embed_url:
        raise SystemExit("EMBED_URL is empty. Set env EMBED_URL")
    if not args.llm_url or not args.llm_model:
        raise SystemExit("LLM_URL / LLM_MODEL is empty. Set env LLM_URL and LLM_MODEL")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.input)
    # Normalize columns
    if "questions" not in df.columns:
        # try common variants
        for cand in ["question", "q", "prompt"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "questions"})
                break
    if "q_id" not in df.columns:
        df["q_id"] = list(range(1, len(df) + 1))

    client = QdrantClient(url=args.qdrant_url)

    answers = []
    pred_answers = []

    system_msg = (
        "你是嚴謹的問答助理，只能根據提供的【檢索內容】回答使用者問題。\n"
        "回答規則：\n"
        "1) 直接回答問題本身，請用 1～3 句完成（越短越好）。\n"
        "2) 不要加入背景知識、延伸說明、額外建議、或與問題無關的內容。\n"
        "3) 若檢索內容不足以回答，請只回：文件未提供足夠資訊。\n"
        "4) 使用與問題相同的語言與用詞。\n"
        "5) 最後加上一行來源：來源：<source>（<source> 來自檢索內容）。"
    )

    for _, row in df.iterrows():
        q_id = row["q_id"]
        q = str(row["questions"])

        q_vec = embed_query(args.embed_url, q)
        ctxs = retrieve_context(client, args.collection, q_vec, top_k=args.top_k, exclude_injection=True)

        ctx_block = format_context_for_prompt(ctxs)
        src = sources_compact(ctxs) if ctxs else ""

        user_msg = f"問題：{q}\n\n【檢索內容】\n{ctx_block if ctx_block else '(無)'}"

        ans = llm_chat(
            llm_url=args.llm_url,
            model=args.llm_model,
            api_key=args.llm_key,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=args.temperature,
        )

        answers.append({"q_id": q_id, "questions": q, "answer": ans, "source": src})
        pred_answers.append({"q_id": q_id, "questions": q, "answer": ans})

        print(f"[{q_id}] done. sources={src}")

    out_df = pd.DataFrame(answers, columns=["q_id", "questions", "answer", "source"])
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote: {args.output}")

    pred_df = pd.DataFrame(pred_answers, columns=["q_id", "questions", "answer"])
    pred_df.to_csv(args.pred_qa_output, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote: {args.pred_qa_output} (for deepeval)")

if __name__ == "__main__":
    main()
