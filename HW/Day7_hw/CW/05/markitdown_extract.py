# markitdown_extract.py
from pathlib import Path

PDF_PATH = Path("example.pdf")
OUT_DIR = Path("output")
OUT_PATH = OUT_DIR / "markitdown.md"

def extract_with_markitdown(pdf_path: Path) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(pdf_path))

    # 兼容不同版本回傳欄位
    for attr in ("text_content", "markdown", "text"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, str) and val.strip():
                return val
    return str(result)

def main():
    print("[RUN] markitdown_extract.py")
    print("[INFO] cwd:", Path.cwd())
    print("[INFO] pdf :", PDF_PATH.resolve())

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"找不到 {PDF_PATH.resolve()}，請確認 example.pdf 是否在 CW/05/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = extract_with_markitdown(PDF_PATH)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] wrote -> {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
