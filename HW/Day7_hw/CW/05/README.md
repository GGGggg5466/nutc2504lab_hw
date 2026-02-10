# 課堂作業 05：PDF 文件轉 Markdown 比較實作

本作業以同一份 `example.pdf` 為輸入，分別使用三種不同工具：
- `pdfplumber`
- `Docling`
- `MarkItDown`

將 PDF 內容擷取並轉換為 Markdown（`.md`）格式，比較三種方法在**文件結構、條列、表格處理**上的差異。

---

## 一、執行環境

- 作業系統：WSL (Linux)
- Python 版本：3.x
- 套件安裝方式：
```bash
pip install pdfplumber docling markitdown
```

## 二、專案結構

CW/05/
├── example.pdf
├── pdfplumber_extract.py
├── docling_extract.py
├── markitdown_extract.py
├── output/
│   ├── pdfplumber.md
│   ├── docling.md
│   └── markitdown.md
└── README.md

## 三、程式寫法說明

1. pdfplumber_extract.py

使用 pdfplumber 逐頁讀取 PDF
每一頁使用 page.extract_text() 抽取純文字
不進行版面或表格語意分析
將結果依頁面順序寫入 Markdown 檔案

特點：

抽字穩定
可完整取得所有文字
理解文件結構（如表格、欄位關係）

2. docling_extract.py

使用 Docling 的 DocumentConverter
設定 PdfPipelineOptions(do_ocr=False)，直接處理可選取文字的 PDF
透過 export_to_markdown() 匯出結果

特點：

能辨識段落與條列語氣
條文閱讀性較佳
表格仍多以文字段落呈現，未完全結構化為 Markdown table

3. markitdown_extract.py

使用 MarkItDown 將 PDF 直接轉為 Markdown
自動處理標題、條列與表格
將 PDF 中的表格轉成標準 Markdown table（| 欄位 |）

特點：

Markdown 結構最完整
表格可直接以 Markdown 形式呈現
最適合後續 LLM / RAG 系統使用

## 四、三種輸出結果差異比較

項目	pdfplumber	Docling	MarkItDown
文字完整度	高	高	高
條列可讀性	低	中	高
標題結構	無	部分保留	明確
表格處理	文字打散	文字化呈現	Markdown table
適合 RAG / LLM	❌	⚠️	✅

## 五、實際觀察說明

pdfplumber.md：

最接近「純文字輸出」
表格內容會被拆成多行文字，欄位關係不明確

docling.md：

條文與段落結構比 pdfplumber 清楚
適合閱讀條款型文件
表格仍未完全轉成結構化格式

markitdown.md：

表格成功轉成 Markdown table
條列、標題結構完整
最接近可直接提供給 LLM 使用的知識格式

## 六、結論

本作業顯示，即使是同一份 PDF，不同文件處理工具在「結構理解能力」上差異明顯：

pdfplumber：適合單純抽取全文文字
Docling：適合需要保留文件語氣與條列的情境
MarkItDown：最適合知識庫、RAG、LLM 等後續應用

## 結果比較（同一份 PDF）

- pdfplumber：抽字穩定，適合純文字內容；但表格容易變成散亂文字。
- Docling：段落與條列可讀性較好；但表格仍可能無法完整結構化。
- MarkItDown：Markdown 結構最好；表格可直接轉成 Markdown table，最適合後續 RAG/LLM 使用。
