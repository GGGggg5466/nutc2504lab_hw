# cw01_step3_get_embeddings.py
# Goal: Get embeddings from senior's embed API (dim=4096) in batches and save to embeddings.json

import json
import time
from typing import Any, Dict, List, Optional

import requests

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
OUT_FILE = "embeddings.json"

TASK_DESCRIPTION = "檢索技術文件"
NORMALIZE = True

# ✅ 批次大小：你目前只有 5 句其實一次送也行，但先照學長要求做批次
BATCH_SIZE = 32

# ✅ (可選) 每批之間稍微喘口氣，避免 API 太敏感（0~0.2 都可以）
SLEEP_SEC = 0.0

texts = [
    "RAG 是 Retrieval-Augmented Generation，用檢索增強生成。",
    "Qdrant 是向量資料庫，可以做相似度搜尋。",
    "Embedding 會把文字轉成向量，讓語意可以被比對。",
    "Top-k 檢索會把最相近的幾段內容取回來當 context。",
    "這是課堂作業 CW01 Step3：用 API 取得向量。",
]


def _chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    """Split list into batches."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def get_embeddings_batch(
    batch_texts: List[str],
    *,
    task_description: Optional[str] = None,
    normalize: bool = True,
    timeout: int = 60,
) -> List[List[float]]:
    """Call embed API for ONE batch and return embeddings."""
    payload: Dict[str, Any] = {
        "texts": batch_texts,
        "normalize": normalize,
    }
    if task_description is not None:
        payload["task_description"] = task_description

    resp = requests.post(EMBED_API_URL, json=payload, timeout=timeout)

    # 不是 2xx 先印出錯誤，debug 比較快
    if not resp.ok:
        print("❌ embed API error:", resp.status_code)
        print(resp.text[:500])
        resp.raise_for_status()

    data = resp.json()
    embeddings = data.get("embeddings") or data.get("embedding")
    if embeddings is None:
        raise ValueError(f"Response missing embeddings field. keys={list(data.keys())}")

    return embeddings


def get_embeddings_batched(
    all_texts: List[str],
    *,
    batch_size: int = 32,
    task_description: Optional[str] = None,
    normalize: bool = True,
    timeout: int = 60,
    sleep_sec: float = 0.0,
) -> List[List[float]]:
    """Call embed API in batches and return embeddings aligned with all_texts."""
    all_embeddings: List[List[float]] = []

    batches = _chunk_list(all_texts, batch_size)
    for idx, batch_texts in enumerate(batches, start=1):
        emb = get_embeddings_batch(
            batch_texts,
            task_description=task_description,
            normalize=normalize,
            timeout=timeout,
        )
        all_embeddings.extend(emb)

        print(f"✅ embed batch {idx}/{len(batches)}  (size={len(batch_texts)})")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # 防呆：確保數量一一對齊
    if len(all_embeddings) != len(all_texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(all_embeddings)} embeddings, "
            f"but have {len(all_texts)} texts."
        )

    return all_embeddings


def main():
    embeddings = get_embeddings_batched(
        texts,
        batch_size=BATCH_SIZE,
        task_description=TASK_DESCRIPTION,
        normalize=NORMALIZE,
        timeout=60,
        sleep_sec=SLEEP_SEC,
    )

    dim = len(embeddings[0])

    payload = {
        "provider": "senior_embed_api",
        "embed_api_url": EMBED_API_URL,
        "task_description": TASK_DESCRIPTION,
        "normalize": NORMALIZE,
        "dim": dim,
        "texts": texts,
        "embeddings": embeddings,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Step3 (senior embed, batched) done")
    print("count:", len(texts))
    print("dim:", dim)
    print("saved:", OUT_FILE)


if __name__ == "__main__":
    main()
