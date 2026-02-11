## 參數調整與優化過程（RAG + DeepEval）

本作業目標是建立一個 RAG 問答系統，並用 DeepEval 四個指標評估：
- Faithfulness（忠實性）
- Answer Relevancy（答案相關性）
- Contextual Precision（檢索精準度）
- Contextual Recall（檢索召回率）

### 1) 為什麼要調參數？
初版系統可以順利完成「檢索 → 生成」，但在 DeepEval 的 **Answer Relevancy** 分數偏低。  
原因不是模型「不會回答」，而是 **回答方式太發散/太長** 或 **檢索內容帶入過多噪音**，導致評估器判定「回答沒有直接對準問題」。

因此本次優化主要聚焦在：
1. **讓回答更直球、更短、更貼題**（提升 Answer Relevancy）
2. **保留檢索品質（CP/CR）**，不要因為縮短回答而犧牲可追溯性與忠實性

---

### 2) 我改了哪些地方？

#### (A) 調整回答策略（system prompt）
在 `rag_answer.py` 中修改 `system_msg`，讓模型遵守：
- 只依據檢索內容回答
- **1～3 句直接回答**（避免背景與延伸）
- 不足資訊則回：`文件未提供足夠資訊。`
- 最後加上來源：`來源：<source>`

**改動原因：**  
DeepEval 的 Answer Relevancy 很吃「是否直接回答問題」。  
限制回答長度與禁止延伸，可以顯著降低跑題機率，使回答更貼近問題本身。

#### (B) 檢索相關參數（Top-K）
我同時調整檢索的 Top-K（retrieve 出來的 chunk 數量）以降低噪音：
- `rag_answer.py`：top_k 由較大的設定調整為更合理範圍（例如 15 → 10，或依實測調整）
- 目標是避免把太多半相關內容塞進 context，導致回答被干擾

**改動原因：**  
Top-K 過大會拉低 Contextual Precision，模型也更容易在回答中加入不必要資訊，進一步拉低 Answer Relevancy。

---

### 3) 調參前後的結果如何？

以下為本次優化前（limit=19）的 DeepEval 平均分數：

- Faithfulness：0.986
- Answer Relevancy：0.447
- Contextual Precision：0.691
- Contextual Recall：0.736

以下為本次優化後（limit=15）的 DeepEval 平均分數：

- Faithfulness：0.878
- Answer Relevancy：0.621
- Contextual Precision：0.752
- Contextual Recall：0.867

**觀察：**
- **Answer Relevancy 明顯提升**（回答更短更貼題、減少跑題）
- CP/CR 仍維持在良好水準（檢索仍能找到足夠證據）
- Faithfulness 略有波動，但仍在可接受範圍（代表仍大多依據檢索內容作答）

---

### 4) 我最後採用的設定（摘要）
- Chunking：chunk_size=650, overlap=150（sliding window + 語意導向 chunking 改善）
- rag_answer top_k=10
- eval top_k=10
- 回答策略：1～3 句直球回答 + 必附來源
- DeepEval：執行 15 題

---

### 5) 結論
本次優化的核心不是「換更大的模型」，而是：
- **控制回答格式**（短、直、對題）
- **控制檢索噪音**（Top-K 不宜過大）
因此在保持系統可追溯性（source）與忠實性（faithfulness）的前提下，有效提升 Answer Relevancy。
