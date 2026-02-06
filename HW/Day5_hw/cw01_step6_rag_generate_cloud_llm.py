# cw01_step6_rag_generate_cloud_llm.py
# RAG Step 6 (Cloud LLM) = Senior Embed API (4096) + Qdrant Retrieval + Senior LLM Generation

import os
import requests

# -----------------------------
# Qdrant (local)
# -----------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw01"
TOP_K = 3

# -----------------------------
# Senior Embedding API (cloud)
# -----------------------------
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
TASK_DESCRIPTION = "檢索技術文件"
NORMALIZE = True

# -----------------------------
# Senior LLM API (cloud)
# -----------------------------
LLM_API_URL = "https://ws-03.wade0426.me/v1/chat/completions"
LLM_MODEL = "/models/gpt-oss-120b"

# If required by your senior's service, set env var:
# export SENIOR_LLM_API_KEY="xxx"
SENIOR_LLM_API_KEY = os.getenv("SENIOR_LLM_API_KEY")  # optional


def embed_senior(texts: list[str]) -> list[list[float]]:
    """Call senior embed API to get embeddings."""
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
    return data["embeddings"]


def retrieve(query_text: str):
    """Retrieve Top-k chunks from Qdrant using senior embeddings (dim=4096)."""
    query_vec = embed_senior([query_text])[0]  # one query -> one vector

    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    payload = {
        "vector": query_vec,
        "limit": TOP_K,
        "with_payload": True,
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", [])


def build_context(results) -> str:
    """Build context string from retrieved results."""
    blocks = []
    used_chunk_ids = []

    for r in results:
        score = r.get("score", 0.0)
        pl = r.get("payload", {}) or {}
        text = pl.get("text", "")
        source = pl.get("source", "")
        chunk_id = pl.get("chunk_id", "")

        used_chunk_ids.append(str(chunk_id))
        blocks.append(f"[source={source} chunk_id={chunk_id} score={score:.4f}]\n{text}")

    context = "\n\n".join(blocks)
    return context, used_chunk_ids


def call_senior_llm(query_text: str, context: str, used_chunk_ids: list[str]) -> str:
    """Call senior LLM API (OpenAI-compatible chat completions)."""
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

    resp = requests.post(LLM_API_URL, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible format:
    return data["choices"][0]["message"]["content"].strip()


def main():
    query_text = "RAG 的核心流程是什麼？"

    results = retrieve(query_text)
    if not results:
        print("❌ No retrieval results. Check Qdrant and collection name.")
        return

    context, used_chunk_ids = build_context(results)
    answer = call_senior_llm(query_text, context, used_chunk_ids)

    print("=== Query ===")
    print(query_text)
    print("\n=== Retrieved Context ===")
    print(context)
    print("\n=== RAG Answer (Senior LLM) ===")
    print(answer)


if __name__ == "__main__":
    print("✅ Step6 done (senior embed + senior LLM)")
    main()
