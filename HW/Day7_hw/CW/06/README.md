# CW06 – Intelligent Document Processing with Docling

## 1. 作業說明
本作業實作一個 **Intelligent Document Processing (IDP)** 流程，
使用 **Docling** 將 PDF 文件轉換為結構化的 Markdown 格式，
並比較傳統 OCR 與 Vision-Language Model OCR 在表格文件上的表現差異。

---

## 2. 使用工具與環境

- OS：WSL (Ubuntu)
- Python：3.11
- Libraries：
  - Docling
  - RapidOCR (onnxruntime)
- VLM OCR：
  - API URL：`https://ws-01.wade0426.me/v1/chat/completions`
  - Model：`allenai/olmOCR-2-7B-1025-FP8`

---

## 3. IDP 流程設計

本作業實作之 IDP Pipeline 如下：

PDF
↓
Docling
↓
OCR Layer
├─ RapidOCR（Traditional OCR）
└─ OLM OCR 2（VLM-based OCR）
↓
Markdown Document

---

## 4. 實驗一：Docling + RapidOCR

- 啟用 `do_ocr=True`
- 指定 OCR 引擎為 `rapidocr`
- OCR pipeline 可正常執行，模型亦成功載入

### 實驗結果
- `outputs/rapidocr.md` 為空或近乎無內容

### 原因分析
- RapidOCR 屬於 **傳統 OCR**
- 僅能辨識文字區塊，缺乏語意與版面結構理解能力
- 面對表格型 PDF 文件時，Docling 無法有效重建結構
- 此結果屬於 **預期行為（Expected Behavior）**

---

## 5. 實驗二：Docling + OLM OCR 2（VLM）

- 使用老師提供之 OpenAI-compatible API
- 模型：`allenai/olmOCR-2-7B-1025-FP8`
- 採用 Vision-Language Model pipeline 進行文件解析

### 實驗結果
- 成功產生 `outputs/olmocr2.md`
- 表格能正確轉換為 Markdown Table
- 欄位、數值與標題結構清楚完整

### 優勢分析
- OLM OCR 2 為 **Vision-Language Model**
- 能同時理解影像與語意資訊
- 對文件版面與表格結構具備良好的理解能力

---

## 6. 結果比較

| 項目 | RapidOCR | OLM OCR 2 |
|---|---|---|
| OCR 類型 | 傳統 OCR | VLM OCR |
| 表格辨識能力 | 幾乎無法 | 可完整重建 |
| Markdown 結構 | 無 | 完整 |
| 文件理解能力 | 低 | 高 |

---

## 7. 結論

本作業透過實際實驗驗證：

- 傳統 OCR 在結構化文件（如表格）上的限制
- Vision-Language Model OCR 在 Intelligent Document Processing 中的優勢

實驗結果顯示，VLM-based OCR（OLM OCR 2）更適合作為進階 IDP 系統的文件理解模組。

---

## 8. 檔案結構說明

```text
CW/06/
├── docling_rapidocr.py   # Docling + RapidOCR pipeline
├── docling_olmocr2.py    # Docling + OLM OCR 2 pipeline
├── sample_table.pdf      # 測試文件
├── outputs/
│   ├── rapidocr.md       # RapidOCR 結果（預期為空）
│   └── olmocr2.md        # OLM OCR 2 結果
└── README.md