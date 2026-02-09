from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable


# ----------------------------
# Tokenizer abstraction
# ----------------------------
class Tokenizer:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    """
    Fallback tokenizer: 用空白切詞來模擬 tokens（不精準，但流程可跑通）。
    encode 回傳「word index」列表，decode 需要原文，因此這個 tokenizer 不支援通用 decode。
    """

    def encode(self, text: str) -> List[int]:
        words = text.split()
        return list(range(len(words)))

    def decode_from_original(self, token_ids: List[int], original_text: str) -> str:
        words = original_text.split()
        picked = [words[i] for i in token_ids if 0 <= i < len(words)]
        return " ".join(picked)

    # 保留 decode 介面，但不建議直接用
    def decode(self, tokens: List[int]) -> str:
        raise RuntimeError("WhitespaceTokenizer.decode() 需要 original_text，請使用 decode_from_original().")


def get_tokenizer(prefer_tiktoken: bool = True) -> Tuple[Tokenizer, str]:
    """
    優先使用 tiktoken(cl100k_base)；若不可用則 fallback 到 whitespace tokenizer。
    """
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
            # 不中斷：改用 fallback
            print(f"[WARN] tiktoken 不可用，改用 whitespace tokenizer。原因：{e!r}")

    return WhitespaceTokenizer(), "whitespace(words)"


# ----------------------------
# Chunking core
# ----------------------------
@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_token: int
    end_token: int  # exclusive
    meta: Dict


def fixed_size_chunk(
    text: str,
    chunk_size: int = 256,
    overlap: int = 48,
    prefer_tiktoken: bool = True,
    meta: Optional[Dict] = None,
) -> Tuple[List[Chunk], str]:
    """
    固定大小分塊（以 tokenizer 的 token 為單位）

    Parameters
    - text: 原始文本
    - chunk_size: 每個 chunk 的 token 數上限
    - overlap: 相鄰 chunk 的重疊 token 數（必須 < chunk_size）
    - prefer_tiktoken: 是否優先用 tiktoken
    - meta: 附加 metadata（例如 source 檔名）

    Returns
    - chunks: Chunk list
    - tokenizer_name: 使用的 tokenizer 名稱
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必須 > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必須 >= 0 且 < chunk_size")

    meta = meta or {}
    tokenizer, tok_name = get_tokenizer(prefer_tiktoken=prefer_tiktoken)

    # tokenize
    if isinstance(tokenizer, WhitespaceTokenizer):
        token_ids = tokenizer.encode(text)
        decode = lambda ids: tokenizer.decode_from_original(ids, text)
    else:
        token_ids = tokenizer.encode(text)
        decode = lambda ids: tokenizer.decode(ids)

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    cid = 0

    for start in range(0, len(token_ids), step):
        end = min(start + chunk_size, len(token_ids))
        piece_ids = token_ids[start:end]
        piece_text = decode(piece_ids)

        chunks.append(
            Chunk(
                chunk_id=cid,
                text=piece_text,
                start_token=start,
                end_token=end,
                meta={**meta, "tokenizer": tok_name, "chunk_size": chunk_size, "overlap": overlap},
            )
        )
        cid += 1
        if end >= len(token_ids):
            break

    return chunks, tok_name


# ----------------------------
# I/O helpers
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
    p = argparse.ArgumentParser(description="Fixed-size Chunking（固定大小分塊）")
    p.add_argument("--input", nargs="+", required=True, help="輸入 txt 檔路徑（可多個）")
    p.add_argument("--chunk-size", type=int, default=256, help="每個 chunk 的 token 上限（預設 256）")
    p.add_argument("--overlap", type=int, default=48, help="chunk 重疊 token 數（預設 48）")
    p.add_argument("--no-tiktoken", action="store_true", help="不要使用 tiktoken，強制用 whitespace tokenizer")
    p.add_argument("--out", type=str, default="", help="輸出 JSONL 檔案路徑（不填則輸出到 stdout）")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    in_paths = [Path(x) for x in args.input]
    prefer_tiktoken = not args.no_tiktoken

    all_records: List[Dict] = []

    for path in in_paths:
        if not path.exists():
            raise FileNotFoundError(f"找不到檔案：{path}")

        text = read_text(path)
        chunks, tok_name = fixed_size_chunk(
            text=text,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            prefer_tiktoken=prefer_tiktoken,
            meta={"source_path": str(path)},
        )

        source_id = path.name
        recs = list(chunks_to_jsonl_records(chunks, source_id=source_id))
        all_records.extend(recs)

        print(f"[OK] {path.name}: {len(chunks)} chunks | tokenizer={tok_name} | chunk_size={args.chunk_size} | overlap={args.overlap}")

    # 輸出 JSONL（每行一個 chunk）
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[DONE] JSONL saved to: {out_path}")
    else:
        # stdout JSONL（只輸出 records；上面的 [OK] 已輸出到 stdout，作業時可自行改成 stderr）
        # 這裡保持簡單：若你要純 JSONL，可把上面的 print 改成 print(..., file=sys.stderr)
        for r in all_records:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()


#固定大小分塊（Fixed-size Chunking）的小工具。
# python "Fixed-size_chunking.py" \
#  --input data_01.txt data_02.txt data_03.txt data_04.txt data_05.txt \
#  --chunk-size 500 --overlap 0 \
#  --out chunks_fixed.jsonl

#滑動窗口分塊（Sliding Window Chunking）的小工具。
# python "Fixed-size_chunking.py" \
#  --input data_01.txt data_02.txt data_03.txt data_04.txt data_05.txt \
#  --chunk-size 256 --overlap 48 \
#  --out chunks_fixed.jsonl