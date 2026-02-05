# Day4 HW — 自動查證 AI（Automatic Fact-Checking Agent）

本作業實作一套 **自動查證 AI 系統**，使用 **LangGraph + LangChain** 建立具備「判斷是否需要搜尋」能力的多節點 Workflow，而非單純一問就搜尋。

---

## 作業目標

- 建立一個具 **Cache / Planner / Search / Final Answer** 的查證流程
- 能判斷 **現有資料是否足夠回答問題**
- 僅在資料不足時，才啟動搜尋補證
- 支援快取，加速重複問題的回應

---

## 系統流程概覽

input
↓
check_cache
├─ Cache Hit → final_answer → end
└─ Cache Miss
↓
planner（評估 knowledge_base 是否足夠）
├─ enough → final_answer → end
└─ not enough
↓
query_gen
↓
search_tool（searXNG）
↓
補充 knowledge_base → 回到 planner


---

## 各節點說明

### 1. check_cache
- 檢查是否已有查證結果
- 命中（Cache Hit）時直接輸出答案

### 2. planner（核心節點）
- 評估目前 `knowledge_base` 是否 **足以完整回答問題**
- 若資料不足，才允許進入搜尋流程
- 此節點負責「是否要搜尋」的決策，而非搜尋本身

### 3. query_gen
- 根據使用者問題，產生適合搜尋引擎的關鍵字

### 4. search_tool
- 使用 searXNG 進行外部搜尋
- 擷取並整理可信來源資料，補充至 knowledge_base

### 5. final_answer
- 彙整 knowledge_base
- 產出具「結論 / 證據摘要 / 不確定性說明」的最終回答
- 將結果寫入 cache，供下次直接使用

---

## 實際查證範例（GTA 6）

### 問題
> **GTA 6 什麼時候上市？延期幾次？**

### 系統行為摘要

1. 第一次詢問：
   - Cache Miss
   - Planner 判定資料不足
   - 進行搜尋（官方公告 + 新聞來源）
   - 補齊 knowledge_base
   - 輸出最終答案並寫入 cache

2. 第二次詢問相同問題：
   - Cache Hit
   - 不再進入 planner / search
   - 直接回傳答案

---

## 關於流程圖「看起來多一條路徑」的說明

- 在 **LangGraph 的 ASCII / Mermaid 圖中**
  - 圖會顯示「所有可能的條件分支」
  - **不代表實際執行時會同時走多條**

- 實際執行時：
  - 每個 conditional edge 只會走其中一條
  - 不存在同時觸發多個分支的情況

### 實際驗證方式

```python
g = app.get_graph()
for e in g.edges:
    print(e)
```

# Day4 HW — LangGraph 進階應用與效能優化

本作業為 LangGraph 進階應用實作，重點在於 **從線性 Chain 架構，進化到具備 Retry、Reflection、人類審核與 Cache 的實務型 Workflow**。  
所有程式皆可於本資料夾直接執行，並可觀察對應的流程行為與輸出結果。

---

## 📁 專案結構


---

## ch6-1：Retry 機制的天氣 API

### 🎯 目標
模擬真實 API 不穩定情境，實作：
- 工具失敗時自動重試（Retry）
- 超過最大次數後進入 fallback
- 避免單一錯誤導致整個流程 Crash

### 🔧 核心設計
- 使用 `ToolNode` 模擬天氣 API
- Router 判斷錯誤次數
- 超過上限自動切換 fallback 節點

### 📌 重點觀察
- 有失敗 → 自動重試
- 成功即返回結果
- 失敗過多 → 明確結束流程

---

## ch6-2：Reflection 機制的翻譯機

### 🎯 目標
實作「**翻譯 → 審查 → 修正 → 再審查**」的循環架構。

### 🔧 核心設計
- `translator`：負責翻譯
- `reflector`：檢查語意是否正確
- Router 判斷：
  - 審查通過（PASS）→ 結束
  - 未通過 → 回到翻譯節點
  - 超過最大嘗試次數 → 強制結束

### 📌 重點觀察
- 能處理諷刺語句、語意反轉
- 每一次修正都有明確理由
- 避免無限迴圈

---

## ch6-3：人工審核的訂單資訊（Human-in-the-loop）

### 🎯 目標
在自動化流程中加入 **人工決策節點**，模擬真實商業系統。

### 🔧 核心設計
- LLM 自動解析訂單資訊（姓名 / 電話 / 商品 / 數量 / 地址）
- 判斷是否為 VIP 客戶
- VIP → 進入人工審核節點
- 管理員輸入 `ok / no` 決定是否通過

### 📌 重點觀察
- 一般客戶：全自動完成
- VIP 客戶：流程暫停，等待人類輸入
- 展現 LangGraph 在高風險流程中的價值

---

## ch7-1：Cache 機制的翻譯機（效能優化）

### 🎯 目標
避免重複呼叫 LLM，提升效能與穩定性。

### 🔧 核心設計
- 翻譯前先檢查 cache
- Cache Hit → 直接回傳結果
- Cache Miss → 呼叫 LLM 並寫入 cache

### 📌 重點觀察
- 第一次翻譯：LLM 執行
- 第二次相同輸入：直接命中快取
- 有效降低 LLM 呼叫次數

---

## ch7-2：混合架構效能優化的 QA Chat

### 🎯 目標
結合三種策略，根據問題特性自動選擇最佳處理路徑。

### 🔧 架構說明
1. **Cache**
   - 相同問題直接回傳結果
2. **Fast Track API**
   - 問候、簡單問題（例如：你好、哈囉）
3. **Expert LLM**
   - 複雜知識型問題（即時串流輸出）

### 📌 Router 判斷邏輯
- Cache Hit → 結束
- 關鍵字屬於簡單問候 → Fast Bot
- 其他問題 → Expert Bot

### 📌 重點觀察
- 同一系統中混合不同模型與策略
- 明確展示效能差異（時間 / 輸出方式）
- Cache 能顯著降低延遲

---

## ✅ 總結

本次 Day4 作業完整展示了 LangGraph 在實務應用中的核心價值：

- 🔁 Retry 與錯誤復原
- 🔄 Reflection 與自我修正
- 👤 Human-in-the-loop 人工決策
- ⚡ Cache 與效能最佳化
- 🧠 多模型、多策略的混合架構

從 Demo 等級的 Chain，進化到可應用於 Production 的 Workflow 設計。

---

## 🛠 執行環境

- Python 3.10+
- LangChain
- LangGraph
- 支援 OpenAI-compatible API 的 LLM Server
