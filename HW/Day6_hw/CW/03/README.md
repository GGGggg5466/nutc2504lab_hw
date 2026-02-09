# 課堂作業 03（CW03）
## Query Rewrite + Retrieval + LLM Answer（RAG Pipeline）

---

## 📌 作業目標

本作業實作一個完整的 **RAG（Retrieval-Augmented Generation）流程**，包含：

1. 將課程提供的文字資料切塊並嵌入至向量資料庫（VDB）
2. 實作 Query Rewrite（問題改寫）
3. 使用改寫後的 Query 進行向量檢索（Retrieval）
4. 結合 LLM 生成回答（Answer Generation）
5. 批次處理 `Re_Write_questions.csv`，並輸出結果為 CSV 檔案

---

## 📂 專案結構

```text
CW/03/
├── CW03_clean.ipynb          # 主程式（整理後可重跑版本）
├── data_01.txt ~ data_05.txt # 課程提供文件資料
├── Re_Write_questions.csv    # 原始問題（依 conversation_id 分組）
├── questions_answer.csv      # 最終輸出結果（本作業產出）
├── Prompt_ReWrite.txt        # Query Rewrite Prompt
├── Prompt.docx               # Answer Generation Prompt
└── README.md
```

# 🧠 系統流程說明

---
## 1️⃣ 文件處理與向量化（VDB）

1. 將 data_*.txt 進行切塊（chunking）
2. 使用 Embedding Model 轉換為向量
3. 將向量與對應的文字內容存入 Qdrant Vector Database

## 2️⃣ Query Rewrite（問題改寫）

1. 使用 LLM 對使用者問題進行改寫
2. 同一個 conversation_id 的問題視為同一段對話
3. 改寫時會考慮前文對話脈絡，使查詢更貼近實際搜尋需求

## 3️⃣ Retrieval（向量檢索）

1. 使用「改寫後的 Query」進行向量搜尋
2. 從 VDB 中取回 Top-K 最相關的文件 chunk
3. chunk 內容作為後續回答的參考依據

## 4️⃣ LLM Answer Generation（回答生成）

1. 將 Retrieval 取得的文件內容組成 Context
2. 呼叫 LLM 產生最終回答
3. 回答同時保留來源資訊（source_file、chunk_id）

## 5️⃣ 批次處理與輸出

1. 讀取 Re_Write_questions.csv
2. 逐筆執行：
- Query Rewrite
- Retrieval
- LLM Answer
3. 將結果寫入新的 CSV 檔案 questions_answer.csv