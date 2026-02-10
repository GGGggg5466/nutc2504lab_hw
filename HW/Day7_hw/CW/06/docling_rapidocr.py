from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pdf_options = PdfPipelineOptions(
    do_ocr=True,            # ✅ 開 OCR
    ocr_engine="rapidocr",  # ✅ 指定 RapidOCR
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
    }
)

result = converter.convert("sample_table.pdf")
md = result.document.export_to_markdown()

print("md length:", len(md))
print("md preview:", md[:200].replace("\n", "\\n"))

Path("outputs").mkdir(exist_ok=True)
with open("outputs/rapidocr.md", "w", encoding="utf-8") as f:
    f.write(md)

print("✅ RapidOCR done -> outputs/rapidocr.md")
