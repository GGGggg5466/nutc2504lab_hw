# cw01_step3_get_embeddings_senior_api.py
# Goal: Get embeddings from senior's embed API (dim=4096) and save to embeddings.json

import json
import requests

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
OUT_FILE = "embeddings.json"

TASK_DESCRIPTION = "檢索技術文件"
NORMALIZE = True

texts = [
    "RAG 是 Retrieval-Augmented Generation，用檢索增強生成。",
    "Qdrant 是向量資料庫，可以做相似度搜尋。",
    "Embedding 會把文字轉成向量，讓語意可以被比對。",
    "Top-k 檢索會把最相近的幾段內容取回來當 context。",
    "這是課堂作業 CW01 Step3：用 API 取得向量。",
]

def main():
    resp = requests.post(
        EMBED_API_URL,
        json={
            "texts": texts,
            "task_description": TASK_DESCRIPTION,
            "normalize": NORMALIZE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    embeddings = data["embeddings"]
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

    print("✅ Step3 (senior embed) done")
    print("count:", len(texts))
    print("dim:", dim)
    print("saved:", OUT_FILE)

if __name__ == "__main__":
    main()
