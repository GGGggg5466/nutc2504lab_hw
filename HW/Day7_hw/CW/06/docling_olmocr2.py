from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from pathlib import Path


def olmocr2_vlm_options():
    return ApiVlmOptions(
        # ✅ 用老師的 HTTPS API
        url="https://ws-01.wade0426.me/v1/chat/completions",
        params={
            "model": "allenai/olmOCR-2-7B-1025-FP8",
            "max_tokens": 4096,
        },
        prompt="Convert this page to clean, readable markdown. Preserve tables as markdown tables.",
        temperature=0.0,
        timeout=120,
        scale=2.0,
        response_format=ResponseFormat.MARKDOWN,
    )


pipeline_options = VlmPipelineOptions(enable_remote_services=True)
pipeline_options.vlm_options = olmocr2_vlm_options()

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
    }
)

result = converter.convert("sample_table.pdf")
md = result.document.export_to_markdown()

Path("outputs").mkdir(exist_ok=True)

with open("outputs/olmocr2.md", "w", encoding="utf-8") as f:
    f.write(md)

print("✅ OLM OCR 2 done -> outputs/olmocr2.md")
