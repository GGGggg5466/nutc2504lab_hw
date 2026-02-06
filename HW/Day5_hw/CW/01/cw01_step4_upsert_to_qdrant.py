# cw01_step4_upsert_to_qdrant.py

import json
from typing import Any, Dict, List, Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw01"
IN_FILE = "embeddings.json"

UPSERT_BATCH_SIZE = 64


def load_embeddings_json(path: str) -> Tuple[int, List[str], List[List[float]], Dict[str, Any]]:
    """Load embeddings.json and validate the required fields."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dim = int(data["dim"])
    texts = data["texts"]
    embeddings = data["embeddings"]

    if len(texts) < 5:
        raise RuntimeError("❌ Need at least 5 texts for CW01 requirement.")
    if len(embeddings) != len(texts):
        raise RuntimeError("❌ embeddings count != texts count")
    if len(embeddings[0]) != dim:
        raise RuntimeError("❌ dim mismatch inside embeddings.json")

    return dim, texts, embeddings, data


def ensure_fresh_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """Create a fresh collection (delete if exists, then create)."""
    if client.collection_exists(collection_name=collection):
        client.delete_collection(collection_name=collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def build_points(
    dim: int,
    texts: List[str],
    embeddings: List[List[float]],
    meta: Dict[str, Any],
) -> List[PointStruct]:
    """Build Qdrant points with payload."""
    model_name = meta.get("model")  # local ST
    provider = meta.get("provider")  # senior API
    embed_api_url = meta.get("embed_api_url")
    task_description = meta.get("task_description")
    normalize = meta.get("normalize")

    points: List[PointStruct] = []
    for i, (t, v) in enumerate(zip(texts, embeddings), start=1):
        payload: Dict[str, Any] = {
            "text": t,
            "source": "cw01_embeddings_json",
            "chunk_id": i,
            "dim": dim,
        }

        # attach whichever metadata exists
        if model_name:
            payload["embedding_model"] = model_name
        if provider:
            payload["embedding_provider"] = provider
        if embed_api_url:
            payload["embed_api_url"] = embed_api_url
        if task_description:
            payload["task_description"] = task_description
        if normalize is not None:
            payload["normalize"] = normalize

        points.append(PointStruct(id=i, vector=v, payload=payload))

    return points


def upsert_points_batched(
    client: QdrantClient,
    collection: str,
    points: List[PointStruct],
    batch_size: int = 64,
) -> None:
    """Upsert points in batches."""
    total = len(points)
    for start in range(0, total, batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection, points=batch)
        print(f"✅ upserted {start + 1}~{start + len(batch)} / {total}")


def main():
    dim, texts, embeddings, meta = load_embeddings_json(IN_FILE)

    client = QdrantClient(url=QDRANT_URL)

    # Fresh collection (avoid deprecated recreate_collection)
    ensure_fresh_collection(client, COLLECTION, dim)

    points = build_points(dim, texts, embeddings, meta)

    # Batch upsert (senior wants batching)
    upsert_points_batched(client, COLLECTION, points, batch_size=UPSERT_BATCH_SIZE)

    info = client.get_collection(COLLECTION)
    print("✅ Step4 done")
    print("collection:", COLLECTION)
    print("dim:", dim)
    print("points_count:", info.points_count)


if __name__ == "__main__":
    main()
