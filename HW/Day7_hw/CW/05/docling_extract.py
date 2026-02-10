# docling_extract.py
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

PDF_PATH = Path("example.pdf")
OUT_DIR = Path("output")
OUT_PATH = OUT_DIR / "docling.md"

def extract_with_docling(pdf_path: Path) -> str:
    pdf_options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()

def main():
    print("[RUN] docling_extract.py")
    print("[INFO] cwd:", Path.cwd())
    print("[INFO] pdf :", PDF_PATH.resolve())

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"找不到 {PDF_PATH.resolve()}，請確認 example.pdf 是否在 CW/05/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = extract_with_docling(PDF_PATH)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] docling -> {OUT_PATH}")

if __name__ == "__main__":
    main()
