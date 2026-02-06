from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw01"
TOP_K = 3

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
TASK_DESCRIPTION = "檢索技術文件"
NORMALIZE = True

EMBED_BATCH_SIZE = 32
EMBED_TIMEOUT = 60
QDRANT_TIMEOUT = 30
SLEEP_SEC = 0.0  # 若 API 容易限流可設 0.05~0.2


def _chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def embed_texts_batched(
    texts: List[str],
    *,
    batch_size: int = 32,
    task_description: Optional[str] = None,
    normalize: bool = True,
    timeout: int = 60,
    sleep_sec: float = 0.0,
) -> List[List[float]]:
    """Embed many texts via senior embed API in batches."""
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


def qdrant_search_rest(
    query_vec: List[float],
    *,
    top_k: int = 3,
    with_payload: bool = True,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """Search in Qdrant via REST API and return result list."""
    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    payload = {"vector": query_vec, "limit": top_k, "with_payload": with_payload}

    resp = requests.post(url, json=payload, timeout=timeout)
    if not resp.ok:
        print(f"❌ qdrant error: status={resp.status_code}")
        print(resp.text[:500])
        resp.raise_for_status()

    data = resp.json()
    return data.get("result", []) or []


def print_results(query_text: str, results: List[Dict[str, Any]], top_k: int) -> None:
    print("✅ Step5 done (senior embed + REST)")
    print("query:", query_text)
    print("top_k:", top_k)
    print("-" * 60)

    if not results:
        print("(no results)")
        return

    for rank, item in enumerate(results, start=1):
        score = float(item.get("score", 0.0))
        pl = item.get("payload", {}) or {}
        text = pl.get("text", "")
        source = pl.get("source", "")
        chunk_id = pl.get("chunk_id", "")

        print(f"[{rank}] score={score:.4f}  source={source}  chunk_id={chunk_id}")
        print(text)
        print("-" * 60)


def main():
    query_text = "RAG 的核心流程是什麼？"

    # ✅ 批次 embedding（即使只有 1 句也照樣走 batch pipeline）
    query_vec = embed_texts_batched(
        [query_text],
        batch_size=EMBED_BATCH_SIZE,
        task_description=TASK_DESCRIPTION,
        normalize=NORMALIZE,
        timeout=EMBED_TIMEOUT,
        sleep_sec=SLEEP_SEC,
    )[0]

    results = qdrant_search_rest(
        query_vec,
        top_k=TOP_K,
        with_payload=True,
        timeout=QDRANT_TIMEOUT,
    )

    print_results(query_text, results, TOP_K)


if __name__ == "__main__":
    main()
