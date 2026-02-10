# pdfplumber_extract.py
from pathlib import Path
import pdfplumber

PDF_PATH = Path("example.pdf")
OUT_DIR = Path("output")
OUT_PATH = OUT_DIR / "pdfplumber.md"

def extract_with_pdfplumber(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            parts.append(f"\n\n---\n\n## Page {i}\n\n{text}")
    return "".join(parts).strip() + "\n"

def main():
    print("[RUN] pdfplumber_extract.py")
    print("[INFO] cwd:", Path.cwd())
    print("[INFO] pdf :", PDF_PATH.resolve())

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"找不到 {PDF_PATH.resolve()}，請確認 example.pdf 是否在 CW/05/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = extract_with_pdfplumber(PDF_PATH)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] wrote -> {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
