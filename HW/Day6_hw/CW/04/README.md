# 課堂作業 04（CW/04）— Hybrid Retrieval + ReRank + LLM 問答系統

## 一、作業目標
本作業目標為建立一套完整的 RAG（Retrieval-Augmented Generation）流程，包含：

1. 將 `.txt` 文件進行語意切塊並建立向量資料庫（VDB）
2. 使用 Hybrid Search（Dense + Sparse / BM25）進行文件檢索
3. 使用 ReRank 模型對檢索結果重新排序
4. 結合大型語言模型（LLM）根據文件內容回答問題
5. 將模型回答與學長提供的標準答案進行對照

---

## 二、系統流程說明

整體流程如下：

- .txt 文件
- 語意切塊（semantic chunking）
- Embedding → Qdrant 向量資料庫
- Hybrid Retrieval（Dense + BM25, RRF 融合）
- ReRank（Qwen3-Reranker-0.6B）
- 組合 Context
- LLM 回答
- 輸出 CSV 與標準答案對照


---

## 三、資料與檔案結構說明

```text
04/
├─ data/
│  ├─ data_01.txt
│  ├─ data_02.txt
│  ├─ data_03.txt
│  ├─ data_04.txt
│  └─ data_05.txt
│
├─ chunks_semantic.jsonl        # 語意切塊後的文件
├─ semantic_chunking.py         # 語意切塊程式
├─ test.ipynb                   # 主程式（Embedding / Retrieval / ReRank / LLM）
├─ Prompt.txt.docx              # LLM Prompt（限制只能依據 context 回答）
├─ questions_answer.csv                # 題目（含學長標準答案與來源文件）
├─ questions_answer_model.csv   # 模型回答結果（含來源文件）
└─ README.md

```
## 四、Hybrid Retrieval 與 ReRank 設計

4.1 Hybrid Search
Dense Retrieval：使用 Embedding API 進行語意向量搜尋
Sparse Retrieval：使用 BM25（Qdrant sparse index）
融合方式：Reciprocal Rank Fusion（RRF）
Hybrid Search 僅負責「找出可能相關的文件」，不直接決定最終排序。

4.2 ReRank

使用本地模型 Qwen3-Reranker-0.6B
對 (Query, Chunk) 配對進行重新排序
取 ReRank 分數最高的 Top-K 文件作為 LLM 的 context

## 五、LLM 回答規則（Prompt 設計）

LLM 只能依據提供的 context 回答
若 context 中無法找到答案，需回覆固定拒答語句
避免模型自行推論或補充文件以外的內容

## 六、批次問答與錯誤處理

在批次處理 questions.csv 時，可能會遇到外部 API（Embedding / LLM）短暫 timeout 的情況，因此程式設計上：
對 API 呼叫加入 重試機制（Retry + Backoff）
若某一題仍失敗，會在「模型回答」欄位標示為 系統錯誤
不影響其他題目繼續處理，確保整體流程可完成

## 七、輸出結果說明

- questions_answer_model.csv

## 八、結論

本作業成功實作一套完整的 Hybrid RAG 系統，結合：
向量檢索（Dense）
關鍵字檢索（Sparse）
ReRank 精修排序
LLM 依文件作答