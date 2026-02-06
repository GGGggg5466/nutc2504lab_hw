import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw01"
IN_FILE = "embeddings.json"

def main():
    # Load embeddings
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required fields
    dim = int(data["dim"])
    texts = data["texts"]
    embeddings = data["embeddings"]

    if len(texts) < 5:
        raise RuntimeError("❌ Need at least 5 texts for CW01 requirement.")
    if len(embeddings) != len(texts):
        raise RuntimeError("❌ embeddings count != texts count")
    if len(embeddings[0]) != dim:
        raise RuntimeError("❌ dim mismatch inside embeddings.json")

    # Optional metadata (compatible)
    model_name = data.get("model")  # local ST
    provider = data.get("provider") # senior API
    embed_api_url = data.get("embed_api_url")
    task_description = data.get("task_description")
    normalize = data.get("normalize")

    # Connect Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # Recreate collection
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Build points
    points = []
    for i, (t, v) in enumerate(zip(texts, embeddings), start=1):
        payload = {
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

    # Upsert
    client.upsert(collection_name=COLLECTION, points=points)

    # Verify
    info = client.get_collection(COLLECTION)
    print("✅ Step4 done")
    print("collection:", COLLECTION)
    print("dim:", dim)
    print("points_count:", info.points_count)

if __name__ == "__main__":
    main()
