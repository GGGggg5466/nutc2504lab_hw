"""
semantic_chunking.py
--------------------
語意分塊（Semantic Chunking）簡單可用版（不依賴外部 Embedding API）。

核心想法
- 先把文本切成「句子」
- 用 TF-IDF（char n-gram）計算相鄰句子相似度（cosine similarity）
- 相似度低於門檻視為主題斷點候選
- 同時遵守 chunk 最大長度（max_tokens），並用 min_tokens 避免切太碎

輸出
- JSONL：每行一個 chunk（含 id/source/chunk_id/start_token/end_token/text/meta）
- 格式與 fixed_size_chunking.py 對齊，方便後續同一套檢索/評分流程

安裝（若缺 sklearn）
    pip install -U scikit-learn

用法
    python semantic_chunking.py --input data_01.txt data_02.txt --max-tokens 256 --min-tokens 80 --sim-threshold 0.12 --out chunks_semantic.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Tokenizer
# ----------------------------
class Tokenizer:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    def encode(self, text: str) -> List[int]:
        words = text.split()
        return list(range(len(words)))

    def decode_from_original(self, token_ids: List[int], original_text: str) -> str:
        words = original_text.split()
        picked = [words[i] for i in token_ids if 0 <= i < len(words)]
        return " ".join(picked)

    def decode(self, tokens: List[int]) -> str:
        raise RuntimeError("WhitespaceTokenizer.decode() 需要 original_text，請使用 decode_from_original().")


def get_tokenizer(prefer_tiktoken: bool = True) -> Tuple[Tokenizer, str]:
    if prefer_tiktoken:
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.get_encoding("cl100k_base")

            class TikTokenizer(Tokenizer):
                def encode(self, text: str) -> List[int]:
                    return enc.encode(text)

                def decode(self, tokens: List[int]) -> str:
                    return enc.decode(tokens)

            return TikTokenizer(), "tiktoken(cl100k_base)"
        except Exception as e:
            print(f"[WARN] tiktoken 不可用，改用 whitespace tokenizer。原因：{e!r}")
    return WhitespaceTokenizer(), "whitespace(words)"


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    if isinstance(tokenizer, WhitespaceTokenizer):
        return len(text.split())
    return len(tokenizer.encode(text))


# ----------------------------
# Sentence splitting (Chinese-friendly)
# ----------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;:])\s*|\n+")

def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


# ----------------------------
# Semantic chunking
# ----------------------------
@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_token: int
    end_token: int
    meta: Dict


def semantic_chunk(
    text: str,
    max_tokens: int = 256,
    min_tokens: int = 80,
    sim_threshold: float = 0.12,
    prefer_tiktoken: bool = True,
    meta: Optional[Dict] = None,
) -> Tuple[List[Chunk], str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens 必須 > 0")
    if min_tokens < 0 or min_tokens >= max_tokens:
        raise ValueError("min_tokens 必須 >=0 且 < max_tokens")

    meta = meta or {}
    tokenizer, tok_name = get_tokenizer(prefer_tiktoken=prefer_tiktoken)

    sents = split_sentences(text)
    if not sents:
        return [], tok_name

    if len(sents) == 1:
        t = sents[0]
        n = count_tokens(t, tokenizer)
        return [Chunk(0, t, 0, n, {**meta, "tokenizer": tok_name, "method": "semantic"})], tok_name

    # 句子向量（中文用 char n-gram TF-IDF 很穩）
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    X = vec.fit_transform(sents)

    # 相鄰相似度：sim[i] = sim(sents[i], sents[i+1])
    sims = cosine_similarity(X[:-1], X[1:]).diagonal().tolist()

    chunks: List[Chunk] = []
    cur: List[str] = []
    cur_tokens = 0
    token_cursor = 0
    chunk_start = 0
    cid = 0

    def flush():
        nonlocal cid, cur, cur_tokens, token_cursor, chunk_start
        if not cur:
            return
        chunk_text = "\n".join(cur).strip()
        chunk_end = chunk_start + cur_tokens
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=chunk_text,
                start_token=chunk_start,
                end_token=chunk_end,
                meta={**meta, "tokenizer": tok_name, "method": "semantic",
                      "max_tokens": max_tokens, "min_tokens": min_tokens, "sim_threshold": sim_threshold},
            )
        )
        cid += 1
        token_cursor = chunk_end
        chunk_start = token_cursor
        cur = []
        cur_tokens = 0

    for i, sent in enumerate(sents):
        t = count_tokens(sent, tokenizer)

        # 單句太長：直接獨立成 chunk
        if t > max_tokens and not cur:
            cur = [sent]
            cur_tokens = t
            flush()
            continue

        # 加上會爆：先收尾
        if cur and (cur_tokens + t) > max_tokens:
            flush()

        if not cur:
            chunk_start = token_cursor

        cur.append(sent)
        cur_tokens += t

        # 若相似度太低 & 已達 min_tokens：切
        if i < len(sims) and sims[i] < sim_threshold and cur_tokens >= min_tokens:
            flush()

    flush()
    return chunks, tok_name


# ----------------------------
# I/O
# ----------------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunks_to_jsonl_records(chunks: List[Chunk], source_id: str) -> Iterable[Dict]:
    for c in chunks:
        yield {
            "id": f"{source_id}::{c.chunk_id}",
            "source": source_id,
            "chunk_id": c.chunk_id,
            "start_token": c.start_token,
            "end_token": c.end_token,
            "text": c.text,
            **c.meta,
        }


# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic Chunking（語意分塊, TF-IDF 近似）")
    p.add_argument("--input", nargs="+", required=True, help="輸入 txt 檔路徑（可多個）")
    p.add_argument("--max-tokens", type=int, default=256, help="chunk 最大 token（預設 256）")
    p.add_argument("--min-tokens", type=int, default=80, help="切點前最小 token（預設 80）")
    p.add_argument("--sim-threshold", type=float, default=0.12, help="相似度低於此值視為斷點（預設 0.12）")
    p.add_argument("--no-tiktoken", action="store_true", help="不要使用 tiktoken，強制用 whitespace tokenizer")
    p.add_argument("--out", type=str, default="", help="輸出 JSONL 檔案路徑（不填則輸出到 stdout）")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    prefer_tiktoken = not args.no_tiktoken
    in_paths = [Path(x) for x in args.input]

    all_records: List[Dict] = []

    for path in in_paths:
        if not path.exists():
            raise FileNotFoundError(f"找不到檔案：{path}")

        text = read_text(path)
        chunks, tok_name = semantic_chunk(
            text=text,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            sim_threshold=args.sim_threshold,
            prefer_tiktoken=prefer_tiktoken,
            meta={"source_path": str(path)},
        )

        recs = list(chunks_to_jsonl_records(chunks, source_id=path.name))
        all_records.extend(recs)

        print(f"[OK] {path.name}: {len(chunks)} chunks | tokenizer={tok_name} | max_tokens={args.max_tokens} | min_tokens={args.min_tokens} | sim_th={args.sim_threshold}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[DONE] JSONL saved to: {out_path}")
    else:
        for r in all_records:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

#python semantic_chunking.py \
#  --input data_01.txt data_02.txt data_03.txt data_04.txt data_05.txt \
#  --max-tokens 256 --min-tokens 80 --sim-threshold 0.12 \
#  --out chunks_semantic.jsonl

#--max-tokens 256：每個 chunk 最多 256 tokens（跟 fixed/sliding 對齊）

#--min-tokens 80：chunk 太短就先不切，避免碎片化

#--sim-threshold 0.12：相鄰句子相似度低於門檻 → 視為主題可能斷掉的切點