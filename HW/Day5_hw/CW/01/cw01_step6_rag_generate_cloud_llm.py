# cw01_step6_rag_generate_cloud_llm.py
# RAG Step 6 (Cloud LLM) = Senior Embed API (4096) + Qdrant Retrieval + Senior LLM Generation
# Style: function-based + batched embedding + REST retrieval (stable)

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# -----------------------------
# Qdrant (local)
# -----------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw01"
TOP_K = 3
QDRANT_TIMEOUT = 30

# -----------------------------
# Senior Embedding API (cloud)
# -----------------------------
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
TASK_DESCRIPTION = "檢索技術文件"
NORMALIZE = True
EMBED_BATCH_SIZE = 32
EMBED_TIMEOUT = 60
SLEEP_SEC = 0.0  # 若 API 容易限流可設 0.05~0.2

# -----------------------------
# Senior LLM API (cloud)
# -----------------------------
LLM_API_URL = "https://ws-03.wade0426.me/v1/chat/completions"
LLM_MODEL = "/models/gpt-oss-120b"
LLM_TIMEOUT = 120

# Optional API key (if required by senior service)
SENIOR_LLM_API_KEY = os.getenv("SENIOR_LLM_API_KEY")  # optional


# -----------------------------
# Utils: batching
# -----------------------------
def _chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# -----------------------------
# Step6-1: Embedding (batched)
# -----------------------------
def embed_texts_batched(
    texts: List[str],
    *,
    batch_size: int = 32,
    task_description: Optional[str] = None,
    normalize: bool = True,
    timeout: int = 60,
    sleep_sec: float = 0.0,
) -> List[List[float]]:
    """Embed texts via senior embed API in batches."""
    all_embeddings: List[List[float]] = []

    batches = _chunk_list(texts, batch_size)
    for idx, batch_texts in enumerate(batches, start=1):
        payload: Dict[str, Any] = {"texts": batch_texts, "normalize": normalize}
        if task_description is not None:
            payload["task_description"] = task_description

        resp = requests.post(EMBED_API_URL, json=payload, timeout=timeout)
        if not resp.ok:
            print(f"❌ embed error: status={resp.status_code}")
            print(resp.text[:500])
            resp.raise_for_status()

        data = resp.json()
        embeddings = data.get("embeddings") or data.get("embedding")
        if embeddings is None:
            raise ValueError(f"Missing embeddings field. keys={list(data.keys())}")

        all_embeddings.extend(embeddings)
        print(f"✅ embed batch {idx}/{len(batches)} (size={len(batch_texts)})")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if len(all_embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(all_embeddings)} != texts {len(texts)}"
        )

    return all_embeddings


def embed_query(query_text: str) -> List[float]:
    """Convenience wrapper: one query -> one vector (still uses batch pipeline)."""
    return embed_texts_batched(
        [query_text],
        batch_size=EMBED_BATCH_SIZE,
        task_description=TASK_DESCRIPTION,
        normalize=NORMALIZE,
        timeout=EMBED_TIMEOUT,
        sleep_sec=SLEEP_SEC,
    )[0]


# -----------------------------
# Step6-2: Retrieve (Qdrant REST)
# -----------------------------
def qdrant_search_rest(
    query_vec: List[float],
    *,
    top_k: int = 3,
    with_payload: bool = True,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    payload = {"vector": query_vec, "limit": top_k, "with_payload": with_payload}

    resp = requests.post(url, json=payload, timeout=timeout)
    if not resp.ok:
        print(f"❌ qdrant error: status={resp.status_code}")
        print(resp.text[:500])
        resp.raise_for_status()

    data = resp.json()
    return data.get("result", []) or []


def build_context(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Build context string and collect used chunk_ids."""
    blocks: List[str] = []
    used_chunk_ids: List[str] = []

    for r in results:
        score = float(r.get("score", 0.0))
        pl = r.get("payload", {}) or {}
        text = pl.get("text", "")
        source = pl.get("source", "")
        chunk_id = pl.get("chunk_id", "")

        used_chunk_ids.append(str(chunk_id))
        blocks.append(f"[source={source} chunk_id={chunk_id} score={score:.4f}]\n{text}")

    return "\n\n".join(blocks), used_chunk_ids


# -----------------------------
# Step6-3: Generate (Senior LLM)
# -----------------------------
def call_senior_llm(query_text: str, context: str, used_chunk_ids: List[str]) -> str:
    system_msg = (
        "你是一個嚴謹的助理。"
        "請只根據【資料】回答【問題】。"
        "如果資料不足以回答，請直接回答：資料不足。"
    )

    user_msg = f"""【資料】
{context}

【問題】
{query_text}

請用繁體中文回答，重點清楚、不要瞎掰。
最後加一行：使用的chunk_id: {", ".join(used_chunk_ids)}
"""

    headers = {"Content-Type": "application/json"}
    if SENIOR_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {SENIOR_LLM_API_KEY}"

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(LLM_API_URL, headers=headers, json=body, timeout=LLM_TIMEOUT)
    if not resp.ok:
        print(f"❌ llm error: status={resp.status_code}")
        print(resp.text[:800])
        resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# -----------------------------
# Orchestration
# -----------------------------
def rag_answer(query_text: str, *, top_k: int = 3) -> Tuple[str, str, List[str]]:
    """Return (context, answer, used_chunk_ids)."""
    qvec = embed_query(query_text)
    results = qdrant_search_rest(qvec, top_k=top_k, with_payload=True, timeout=QDRANT_TIMEOUT)
    if not results:
        raise RuntimeError("No retrieval results. Check Qdrant container and collection name.")
    context, used_chunk_ids = build_context(results)
    answer = call_senior_llm(query_text, context, used_chunk_ids)
    return context, answer, used_chunk_ids


def main():
    query_text = "RAG 的核心流程是什麼？"
    context, answer, used_chunk_ids = rag_answer(query_text, top_k=TOP_K)

    print("✅ Step6 done (senior embed + senior LLM)")
    print("\n=== Query ===")
    print(query_text)
    print("\n=== Retrieved Context ===")
    print(context)
    print("\n=== RAG Answer (Senior LLM) ===")
    print(answer)
    print("\n(used_chunk_id)", ", ".join(used_chunk_ids))


if __name__ == "__main__":
    main()
