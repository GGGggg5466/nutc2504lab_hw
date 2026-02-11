# Day5 RAG 作業（Embedding + Qdrant）

本作業實作一個完整的 **Retrieval-Augmented Generation (RAG)** 流程，  
包含三種文本切塊方法、Embedding、向量資料庫 Qdrant 檢索，以及透過 API 自動評分，  
並比較不同切塊策略對檢索效果的影響。

---

## 📂 專案結構

Homework/
├── data_01.txt ~ data_05.txt # 原始文本資料
├── questions.csv # 問題集（共 20 題）
│
├── chunks_fixed.jsonl # 固定大小切塊結果
├── chunks_sliding.jsonl # 滑動視窗切塊結果
├── chunks_semantic.jsonl # 語意切塊結果
│
├── day5_index_qdrant.py # 將三種 chunks 建立向量索引至 Qdrant
│
├── s1411232035_RAG_HW_01.py # 主程式（檢索 + API 評分 + 產生 CSV）
├── s1411232035_RAG_HW_01.csv # 最終作業結果（60 筆）
├── s1411232035_RAG_HW_01.pdf # 作業第 4 題說明文件
└── README.md


---

## ✂️ 文本切塊方法說明

本作業比較以下三種切塊策略：

### 1️⃣ 固定大小切塊（Fixed-size Chunking）
- `chunk_size = 512`
- `overlap = 64`
- 每個 chunk 長度固定，不考慮語意邊界

### 2️⃣ 滑動視窗切塊（Sliding Window）
- `chunk_size = 384`
- `overlap = 128`
- 相鄰 chunk 保留部分重疊文字，提高上下文連貫性

### 3️⃣ 語意切塊（Semantic Chunking）
- `max_tokens = 512`
- `min_tokens = 84`
- `similarity_threshold = 0.12`
- 依據 embedding 相似度動態決定切塊邊界

---

## 🧠 RAG 系統流程

1. 將三種切塊結果進行 Embedding
2. 將向量資料存入 **Qdrant** 向量資料庫
3. 對每一題問題進行：
   - 問題 embedding
   - 向量相似度搜尋（Top-1）
   - 取回最相關文本作為 `retrieve_text`
4. 將檢索結果送至評分 API，取得分數
5. 將結果整理成 CSV 檔案

---

## 📊 作業輸出說明（CSV）

`s1411232035_RAG_HW_01.csv` 共 **60 筆資料（20 題 × 3 種方法）**

欄位定義如下：

| 欄位名稱 | 說明 |
|--------|------|
| id | 唯一編號 |
| q_id | 題號 |
| method | 切塊方法（fixed / sliding / semantic） |
| retrieve_text | 檢索到的文本內容 |
| score | API 回傳的評分 |
| source | 文本來源檔案 |

---

## 📈 實驗結果摘要

比較三種切塊方式的平均分數：

- **語意切塊（Semantic Chunking）平均分最高**
- 其次為滑動視窗切塊
- 固定大小切塊分數最低

顯示考慮語意邊界的切塊方式，能提供較高品質的檢索結果。

---

## ▶️ 執行方式

### 1️⃣ 建立向量索引（需先啟動 Qdrant）
```bash
python day5_index_qdrant.py
