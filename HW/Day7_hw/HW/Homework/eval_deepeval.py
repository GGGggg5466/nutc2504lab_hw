#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,re
import json
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM


# =========================
# Custom evaluator LLM for DeepEval
# =========================

# Ensure these imports exist at the top of eval_deepeval.py
# import requests
# import json
# import re
# from typing import Any, Dict, Optional
# from deepeval.models import DeepEvalBaseLLM

class OpenAICompatLLM(DeepEvalBaseLLM):

    def __init__(self, llm_url: str, model: str, api_key: str = ""):
        self.llm_url = llm_url
        self.model = model
        self.api_key = api_key
        self._loaded = False

    def load_model(self):
        # For HTTP-based LLM, "loading" just means marking it ready.
        self._loaded = True
        return self

    def get_model_name(self) -> str:
        return self.model

    @staticmethod
    def _extract_json(text: str):
        """
        Best-effort extraction/parsing of JSON object/array from model output.
        Returns parsed Python object if possible, else None.
        """
        if not text:
            return None
        t = text.strip()

        # 1) direct parse
        try:
            return json.loads(t)
        except Exception:
            pass

        # 2) strip markdown fences
        t2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            return json.loads(t2)
        except Exception:
            pass

        # 3) find first {...} or [...]
        m = re.search(r"(\{.*\}|\[.*\])", t2, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None

        return None

    def _post_chat(self, system_msg: str, user_msg: str, timeout: int = 180) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
        }

        resp = requests.post(self.llm_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-like response
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            c0 = data["choices"][0]
            if isinstance(c0, dict):
                if "message" in c0 and c0["message"]:
                    return (c0["message"].get("content") or "").strip()
                if "text" in c0:
                    return (c0.get("text") or "").strip()

        raise ValueError(f"Unexpected evaluator response: {data}")

    def generate(self, prompt: str) -> str:
        if not self._loaded:
            self.load_model()

        # Force schema-compliant JSON only (important for AnswerRelevancy, etc.)
        system_json_only = (
            "You are an evaluation engine. "
            "Return ONLY valid JSON that strictly matches the required schema. "
            "No extra text, no markdown, no explanations. "
            "Use the exact key names required by the schema (e.g., 'verdict', 'reason', etc.)."
        )

        # 1st attempt
        out = self._post_chat(system_json_only, prompt)

        # Validate JSON; if not valid, repair once
        if self._extract_json(out) is None:
            repair_prompt = (
                "Your previous response was NOT valid JSON.\n"
                "Rewrite the answer as ONLY valid JSON that strictly matches the required schema.\n"
                "Do NOT add any other text.\n\n"
                f"TASK:\n{prompt}\n\n"
                f"BAD OUTPUT:\n{out}"
            )
            out2 = self._post_chat(system_json_only, repair_prompt)
            obj2 = self._extract_json(out2)
            if obj2 is not None:
                obj2 = self._normalize_deepeval_schema(obj2)
                return json.dumps(obj2, ensure_ascii=False)
            return out2.strip()

        obj = self._extract_json(out)
        if obj is not None:
            obj = self._normalize_deepeval_schema(obj)
            return json.dumps(obj, ensure_ascii=False)
        return out.strip()
    
    @staticmethod
    def _normalize_deepeval_schema(obj):
        """
        DeepEval AnswerRelevancyMetric expects:
          {"verdicts": [{"verdict": "...", "reason": "..."} , ...]}
        Some models return verdict items missing 'verdict'.
        We patch it to avoid Pydantic ValidationError.
        """
        if not isinstance(obj, dict):
            return obj

        v = obj.get("verdicts", None)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    # Fill missing keys
                    if "verdict" not in it:
                        it["verdict"] = "no"
                    if "reason" not in it:
                        it["reason"] = ""
        return obj


    async def a_generate(self, prompt: str) -> str:
        # DeepEval async mode calls this
        return self.generate(prompt)


# =========================
# Retrieval for contexts (evaluate needs contexts)
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


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15, help="Only evaluate first N questions (default: 3)")
    ap.add_argument("--pred", default="outputs/pred_questions_answer.csv", help="Pred CSV: q_id,questions,answer")
    ap.add_argument("--gold", default="data/questions_answer.csv", help="Gold CSV: q_id,questions,answer (or similar)")
    ap.add_argument("--report", default="outputs/deepeval_report.json", help="JSON report output")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--collection", default="day7_collection_v2")
    ap.add_argument("--embed_url", default="https://ws-04.wade0426.me/embed")
    ap.add_argument("--eval_llm_url", default="https://ws-05.huannago.com/v1/chat/completions")
    ap.add_argument("--eval_llm_model", default="Qwen3-VL-8B-Instruct-BF16.gguf")
    ap.add_argument("--eval_llm_key", default="")
    ap.add_argument("--top_k", type=int, default=int(os.getenv("EVAL_TOP_K", "10")))
    args = ap.parse_args()

    if not args.embed_url:
        raise SystemExit("EMBED_URL is empty. Set env EMBED_URL")
    if not args.eval_llm_url or not args.eval_llm_model:
        raise SystemExit("EVAL_LLM_URL / EVAL_LLM_MODEL is empty. Set env EVAL_LLM_URL/EVAL_LLM_MODEL (or reuse LLM_URL/LLM_MODEL)")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    pred_df = pd.read_csv(args.pred)
    gold_df = pd.read_csv(args.gold)

    # Normalize column names
    if "questions" not in pred_df.columns:
        for cand in ["question", "q", "prompt"]:
            if cand in pred_df.columns:
                pred_df = pred_df.rename(columns={cand: "questions"})
                break
    if "questions" not in gold_df.columns:
        for cand in ["question", "q", "prompt"]:
            if cand in gold_df.columns:
                gold_df = gold_df.rename(columns={cand: "questions"})
                break

    if "q_id" not in pred_df.columns:
        pred_df["q_id"] = list(range(1, len(pred_df) + 1))
    if "q_id" not in gold_df.columns:
        gold_df["q_id"] = list(range(1, len(gold_df) + 1))

    # Join by q_id (preferred), fallback by questions
    merged = pd.merge(pred_df, gold_df, on="q_id", how="inner", suffixes=("_pred", "_gold"))
    if merged.empty:
        merged = pd.merge(pred_df, gold_df, on="questions", how="inner", suffixes=("_pred", "_gold"))
    if merged.empty:
        raise SystemExit("Cannot align pred and gold. Ensure q_id matches or questions text matches.")

    client = QdrantClient(url=args.qdrant_url)

    eval_llm = OpenAICompatLLM(args.eval_llm_url, args.eval_llm_model, args.eval_llm_key)

    faithfulness = FaithfulnessMetric(model=eval_llm)
    relevancy = AnswerRelevancyMetric(model=eval_llm)
    c_precision = ContextualPrecisionMetric(model=eval_llm)
    c_recall = ContextualRecallMetric(model=eval_llm)

    rows_out = []
    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_precision": [],
        "contextual_recall": [],
    }

    merged = merged.head(args.limit)
    print(f"[INFO] Evaluating {len(merged)} questions (limit={args.limit})")

    for _, r in merged.iterrows():
        qid = int(r["q_id"])
        question = str(r["questions_pred"]) if "questions_pred" in r else str(r["questions"])
        pred_ans = str(r["answer_pred"])
        gold_ans = str(r["answer_gold"])

        # Re-retrieve contexts so metrics can score grounding
        q_vec = embed_query(args.embed_url, question)
        ctxs_raw = retrieve_context(client, args.collection, q_vec, top_k=args.top_k, exclude_injection=True)

        # ---- normalize retrieval_context to List[str] for DeepEval ----
        ctxs: List[str] = []
        for c in (ctxs_raw or []):
            if isinstance(c, str):
                t = c.strip()
            elif isinstance(c, dict):
                t = str(c.get("text", "")).strip()
            else:
                t = str(c).strip()
            if t:
                ctxs.append(t)

        tc = LLMTestCase(
            input=question,
            actual_output=pred_ans,
            expected_output=gold_ans,
            retrieval_context=ctxs,   # 반드시 List[str]
        )

        # Measure
        faithfulness.measure(tc)
        relevancy.measure(tc)
        c_precision.measure(tc)
        c_recall.measure(tc)

        row = {
            "q_id": qid,
            "faithfulness": float(faithfulness.score or 0.0),
            "answer_relevancy": float(relevancy.score or 0.0),
            "contextual_precision": float(c_precision.score or 0.0),
            "contextual_recall": float(c_recall.score or 0.0),
        }
        rows_out.append(row)
        scores["faithfulness"].append(row["faithfulness"])
        scores["answer_relevancy"].append(row["answer_relevancy"])
        scores["contextual_precision"].append(row["contextual_precision"])
        scores["contextual_recall"].append(row["contextual_recall"])

        print(f"[{qid}] f={row['faithfulness']:.3f} r={row['answer_relevancy']:.3f} "
              f"cp={row['contextual_precision']:.3f} cr={row['contextual_recall']:.3f}")

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    report = {
        "count": len(rows_out),
        "averages": {k: avg(v) for k, v in scores.items()},
        "per_item": rows_out,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote report: {args.report}")
    print("[Averages]", report["averages"])

if __name__ == "__main__":
    main()
