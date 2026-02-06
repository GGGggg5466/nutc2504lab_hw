# cw01_step5_search_from_qdrant_senior_embed.py
# Goal: Use senior embed API for query -> 4096 vector, then Qdrant REST search.

import requests

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw01"
TOP_K = 3

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
TASK_DESCRIPTION = "檢索技術文件"
NORMALIZE = True

def embed_text(text: str):
    resp = requests.post(
        EMBED_API_URL,
        json={"texts": [text], "task_description": TASK_DESCRIPTION, "normalize": NORMALIZE},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def main():
    query_text = "RAG 的核心流程是什麼？"
    query_vec = embed_text(query_text)

    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    resp = requests.post(
        url,
        json={"vector": query_vec, "limit": TOP_K, "with_payload": True},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("result", [])

    print("✅ Step5 done (senior embed)")
    print("query:", query_text)
    print("top_k:", TOP_K)
    print("-" * 60)

    for rank, item in enumerate(results, start=1):
        score = item.get("score", 0.0)
        pl = item.get("payload", {}) or {}
        text = pl.get("text", "")
        source = pl.get("source", "")
        chunk_id = pl.get("chunk_id", "")
        print(f"[{rank}] score={score:.4f}  source={source}  chunk_id={chunk_id}")
        print(text)
        print("-" * 60)

if __name__ == "__main__":
    main()
