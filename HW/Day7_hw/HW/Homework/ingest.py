#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import uuid
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document
import requests
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# =========================
# Config / Helpers
# =========================

INJECTION_PATTERNS = [
    r"請忽略.*系統指令",
    r"請忽略.*所有系統指示",
    r"Please ignore.*system instructions",
    r"Please ignore.*system prompts",
    r"From now on, you are",
    r"現在開始你是",
    r"你是一位.*(老師|甜點師傅)",
    r"tiramisu is delicious",
    r"提拉米蘇很好吃",
]

INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), flags=re.IGNORECASE)

def is_injection_text(text: str) -> bool:
    if not text:
        return False
    return bool(INJECTION_RE.search(text))

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 650, overlap: int = 150) -> List[str]:
    """
    Semantic-ish chunking for FAQ/legal docs.
    Priority:
      1) Split by Q1/Q2/... blocks if present (best for your dataset)
      2) Otherwise split by paragraphs (blank lines)
    Then pack blocks into chunks <= chunk_size, with overlap (by tail text).
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    # Keep paragraph boundaries: normalize_whitespace already keeps \n\n
    raw = text

    # ---- 1) Try split by Qn blocks (FAQ style) ----
    # Match start of a question marker like "Q1", "Q2", possibly with spaces/punctuations
    q_pat = re.compile(r"(?:^|\n)(Q\s*\d+)\b", flags=re.IGNORECASE)
    q_matches = list(q_pat.finditer(raw))

    blocks: List[str] = []
    if len(q_matches) >= 2:
        for i, m in enumerate(q_matches):
            start = m.start()
            end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(raw)
            blk = raw[start:end].strip()
            if blk:
                blocks.append(blk)
    else:
        # ---- 2) Fallback: paragraph blocks ----
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        blocks = paras if paras else [raw]

    # ---- 3) Pack blocks into chunks with a soft overlap ----
    chunks: List[str] = []
    cur = ""

    def flush_with_overlap(cur_text: str):
        cur_text = cur_text.strip()
        if not cur_text:
            return ""
        chunks.append(cur_text)

        # Overlap: carry tail text to next chunk (but avoid huge carry)
        if overlap <= 0:
            return ""
        tail = cur_text[-overlap:].strip()
        return tail

    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue

        # If a single block is extremely long, do a small sliding cut inside it
        if len(blk) > chunk_size * 2:
            # flush current first
            if cur:
                carry = flush_with_overlap(cur)
                cur = carry
            # cut the long block by char sliding (backup)
            i = 0
            n = len(blk)
            step = max(chunk_size - overlap, 1)
            while i < n:
                part = blk[i:i + chunk_size].strip()
                if part:
                    chunks.append(part)
                i += step
            cur = ""
            continue

        # Pack normally
        if not cur:
            cur = blk
        else:
            tentative = cur + "\n\n" + blk
            if len(tentative) <= chunk_size:
                cur = tentative
            else:
                carry = flush_with_overlap(cur)
                # start next chunk with overlap tail + new block
                if carry:
                    # keep overlap separated to avoid gluing words
                    cur = carry + "\n\n" + blk
                else:
                    cur = blk

                # If still too big (rare), hard cut once
                if len(cur) > chunk_size * 1.2:
                    carry2 = flush_with_overlap(cur[:chunk_size])
                    cur = carry2 + cur[chunk_size:]

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


@dataclass
class DocChunk:
    id: str
    text: str
    source: str
    page: Optional[int]
    is_injection: bool


# =========================
# OCR / Extraction (IDP)
# =========================

def extract_text_from_pdf(pdf_path: str, ocr_lang: str = "chi_tra+eng") -> List[Tuple[Optional[int], str]]:
    """
    Return list of (page_number_1based, text).
    Strategy:
      - Try text extraction via pdfplumber per page.
      - If extracted text is too small, render page image + OCR.
    """
    results: List[Tuple[Optional[int], str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            page_no = idx + 1
            txt = page.extract_text() or ""
            txt = normalize_whitespace(txt)

            # If not enough text, OCR that page
            if len(txt) < 30:
                txt = ocr_pdf_page(pdf_path, page_index=idx, lang=ocr_lang)

            results.append((page_no, txt))
    return results

def ocr_pdf_page(pdf_path: str, page_index: int, lang: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        # 2x zoom for better OCR
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang=lang)
        return normalize_whitespace(text)
    finally:
        doc.close()

def extract_text_from_png(png_path: str, ocr_lang: str = "chi_tra+eng") -> str:
    img = Image.open(png_path)
    text = pytesseract.image_to_string(img, lang=ocr_lang)
    return normalize_whitespace(text)

def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return normalize_whitespace("\n".join(parts))


# =========================
# Embedding API
# =========================

def embed_texts(embed_url: str, texts: List[str], task_description: str = "檢索技術文件", normalize: bool = True, timeout: int = 120) -> List[List[float]]:
    """
    Calls your embed endpoint:
      POST { embed_url } with json:
        { "texts": [...], "task_description": "...", "normalize": true }
    Expected response:
      - Either {"embeddings": [[...], ...]}
      - Or directly [[...], ...]
    """
    resp = requests.post(
        embed_url,
        json={"texts": texts, "task_description": task_description, "normalize": normalize},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected embed response format: {type(data)} keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}")


# =========================
# Qdrant
# =========================

def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )

def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: List[DocChunk],
    vectors: List[List[float]],
) -> None:
    assert len(chunks) == len(vectors)
    points = []
    for ch, vec in zip(chunks, vectors):
        payload = {
            "text": ch.text,
            "source": ch.source,
            "page": ch.page,
            "is_injection": ch.is_injection,
        }
        points.append(
            qmodels.PointStruct(
                id=ch.id,
                vector=vec,
                payload=payload,
            )
        )
    client.upsert(collection_name=collection, points=points)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Folder containing pdf/png/docx")
    ap.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "day7_collection_v2"))
    ap.add_argument("--embed_url", default=os.getenv("EMBED_URL", "https://ws-04.wade0426.me/embed"))
    ap.add_argument("--ocr_lang", default=os.getenv("OCR_LANG", "chi_tra+eng"))
    ap.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "900")))
    ap.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "150")))
    ap.add_argument("--batch_size", type=int, default=int(os.getenv("EMBED_BATCH", "32")))
    args = ap.parse_args()

    if not args.embed_url:
        raise SystemExit("EMBED_URL is empty. Set env EMBED_URL or pass --embed_url")

    data_dir = args.data_dir
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".pdf", ".png", ".docx"))])

    if not files:
        raise SystemExit(f"No pdf/png/docx found under: {data_dir}")

    client = QdrantClient(url=args.qdrant_url)

    all_chunks: List[DocChunk] = []

    # Extract + chunk
    for fn in files:
        path = os.path.join(data_dir, fn)
        ext = fn.lower().split(".")[-1]

        if ext == "pdf":
            pages = extract_text_from_pdf(path, ocr_lang=args.ocr_lang)
            for page_no, text in pages:
                for c in chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap):
                    ch = DocChunk(
                        id=str(uuid.uuid4()),
                        text=c,
                        source=fn,
                        page=page_no,
                        is_injection=is_injection_text(c),
                    )
                    all_chunks.append(ch)

        elif ext == "png":
            text = extract_text_from_png(path, ocr_lang=args.ocr_lang)
            for c in chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap):
                ch = DocChunk(
                    id=str(uuid.uuid4()),
                    text=c,
                    source=fn,
                    page=None,
                    is_injection=is_injection_text(c),
                )
                all_chunks.append(ch)

        elif ext == "docx":
            text = extract_text_from_docx(path)
            for c in chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap):
                ch = DocChunk(
                    id=str(uuid.uuid4()),
                    text=c,
                    source=fn,
                    page=None,
                    is_injection=is_injection_text(c),
                )
                all_chunks.append(ch)

    if not all_chunks:
        raise SystemExit("No chunks produced. Check OCR/text extraction.")

    # Determine embedding dim by embedding 1 chunk
    probe_vec = embed_texts(args.embed_url, [all_chunks[0].text])[0]
    vector_size = len(probe_vec)

    ensure_collection(client, args.collection, vector_size)

    # Batch embed + upsert
    bs = args.batch_size
    for i in tqdm(range(0, len(all_chunks), bs), desc="Embedding+Upserting"):
        batch = all_chunks[i:i+bs]
        vecs = embed_texts(args.embed_url, [c.text for c in batch])
        upsert_chunks(client, args.collection, batch, vecs)

    # Summary
    inj_count = sum(1 for c in all_chunks if c.is_injection)
    print(f"[OK] Ingested {len(all_chunks)} chunks into Qdrant collection='{args.collection}'.")
    print(f"[OK] Detected injection-like chunks: {inj_count}")
    if inj_count:
        suspects = sorted(set([c.source for c in all_chunks if c.is_injection]))
        print(f"[Suspect sources] {suspects}")

if __name__ == "__main__":
    main()
