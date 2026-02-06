import json
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = "http://localhost:6333"
EMBED_URL = "https://ws-04.wade0426.me/embed"
TASK_DESC = "檢索技術文件"
DIM = 4096

def embed_texts(texts):
    r = requests.post(
        EMBED_URL,
        json={"texts": texts, "task_description": TASK_DESC, "normalize": True},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["embeddings"]

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def ensure_collection(client: QdrantClient, name: str):
    # 如果已存在就跳過；不存在就建
    try:
        client.get_collection(name)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

def index_jsonl_to_collection(jsonl_path: str, collection: str, method_name: str, batch_size=32):
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, collection)

    chunks = load_jsonl(jsonl_path)

    ids, vectors, payloads = [], [], []
    for i, ch in enumerate(chunks):
        text = ch.get("text", "")
        if not text.strip():
            continue

        ids.append(i)
        payloads.append({
            "text": text,
            "source": ch.get("source", ""),
            "chunk_id": ch.get("chunk_id", i),
            "method": method_name,
        })

    # 批次 embedding + upsert
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_payloads = payloads[start:end]
        batch_texts = [p["text"] for p in batch_payloads]

        batch_vecs = embed_texts(batch_texts)

        client.upsert(
            collection_name=collection,
            points=[
                {"id": int(pid), "vector": vec, "payload": pay}
                for pid, vec, pay in zip(batch_ids, batch_vecs, batch_payloads)
            ],
        )
        print(f"[OK] upsert {collection}: {end if end < len(ids) else len(ids)}/{len(ids)}")

if __name__ == "__main__":
    # 依你實際檔名調整
    index_jsonl_to_collection("chunks_fixed.jsonl",   "day5_fixed",   "固定大小")
    index_jsonl_to_collection("chunks_sliding.jsonl", "day5_sliding", "滑動視窗")
    index_jsonl_to_collection("chunks_semantic.jsonl","day5_semantic","語意切塊")
    print("[DONE] indexing all collections")
