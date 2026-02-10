# Day6 HW — RAG + Re-ranker + DeepEval 評估

本作業實作一個 **Retrieval-Augmented Generation (RAG)** 問答流程，結合  
**Hybrid Retrieval（Dense + Sparse + RRF）**、**Qwen3-Reranker-0.6B** 重排序，
並使用 **DeepEval** 對回答品質進行五項指標評估。

本次評估先以 **5 題**進行測試，確認整體流程正確與指標可正常產生數值。

---

## 一、系統流程總覽

- User Question
- Query Rewrite（LLM）
- Hybrid Retrieval（Dense + BM25 → RRF）
- Qwen3-Reranker-0.6B（Top-k 重排序）
- RAG Answering（LLM 僅使用 contexts）
- DeepEval（5 個評估指標）


---

## 二、Retrieval 與 Re-rank 設計

### 1. Hybrid Retrieval（Qdrant Fusion Query）
- **Dense Retrieval**：語意向量相似度
- **Sparse Retrieval（BM25）**：關鍵字匹配
- **RRF（Reciprocal Rank Fusion）**：
  - 融合 Dense / Sparse 排名
  - 提升整體召回率與穩定度

### 2. Re-ranker
- 模型：`Qwen3-Reranker-0.6B`
- 功能：對候選 contexts 進行精細排序
- 輸出：依相關性排序的 Top-k contexts

---

## 三、RAG Answering 設計

- 僅允許 LLM 使用 **re-rank 後的 contexts**
- 若 contexts 不足，需明確回覆：
  >「參考資料沒有提到，無法回答此問題」
- 目的：避免 hallucination，提高 Faithfulness

---

## 四、DeepEval 評估指標說明

本作業使用 **五個 DeepEval 指標**：

| 指標 | 說明 |
|---|---|
| **Faithfulness** | 回答是否忠實依據 contexts（是否胡編） |
| **Answer Relevancy** | 回答是否真正回應問題 |
| **Contextual Recall** | contexts 是否涵蓋回答所需資訊 |
| **Contextual Precision** | contexts 是否精準、不冗餘 |
| **Contextual Relevancy** | contexts 與問題的整體相關性 |

> 指標範圍：`0 ~ 1`，數值越高代表表現越好

---

## 五、實驗設定

- 評估題數：**5 題**
- Re-rank Top-k：`k = 5`
- Judge LLM：Remote LLM（JSON-only 輸出）
- 評估方式：
  - 每一題獨立計算五項指標
  - 避免單一題目錯誤導致整體中斷

---

## 六、評估結果摘要（前 5 題）

| q_id | Faithfulness | Answer Relevancy | Contextual Recall | Contextual Precision | Contextual Relevancy |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.857 | 1.00 | 0.804 | 0.40 |
| 2 | 1.00 | 0.667 | 1.00 | 0.75 | 0.27 |
| 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.41 |
| 4 | 1.00 | 0.667 | 1.00 | 0.887 | 0.56 |
| 5 | 1.00 | 1.00 | 1.00 | 0.50 | 0.20 |

---

## 七、結果分析

### 1. Faithfulness
- **全數為 1.0**
- 代表回答皆嚴格依據檢索到的資料，無 hallucination

### 2. Answer Relevancy
- 多數介於 `0.66 ~ 1.0`
- 顯示回答大致能切中問題重點
- 部分問題因資訊限制，相關度略低

### 3. Contextual Recall
- **全數為 1.0**
- Hybrid Retrieval + RRF 能穩定找出涵蓋答案的資料

### 4. Contextual Precision
- 約落在 `0.5 ~ 1.0`
- 顯示 contexts 偶有冗餘資訊
- 未來可透過：
  - 降低 top-k
  - 或在 re-rank 後做 context trimming 改善

### 5. Contextual Relevancy
- 數值偏低（0.2 ~ 0.56）
- 主因為：
  - 官方 FAQ 類文件內容廣泛
  - 與特定問題的專一性有限

---

## 八、結論

- 本次成功完成：
  - **RAG Pipeline**
  - **Qwen3-Reranker 重排序**
  - **DeepEval 五指標量化評估**
- 指標結果合理、可解釋，流程穩定
- 已具備擴展至 **30 題完整評估** 的基礎

---

## 九、未來改進方向

- 調整 re-rank top-k 以提升 Contextual Precision
- 增加 chunk-level filtering
- 比較不同 reranker / embedding 模型對指標的影響
- 針對 Contextual Relevancy 設計更專一的資料切分策略

---

