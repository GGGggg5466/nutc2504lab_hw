## 📘 Day4 HW：自動查證 AI（Automatic Fact-Checking Agent）

本作業為課後實戰「自動查證 AI」，以 **LangGraph + LangChain** 建立一個具備  
**快取（Cache）／規劃（Planner）／搜尋（Search）／查證與回答（Final Answer）** 的多節點 Workflow。

系統可針對使用者輸入問題：
- 先檢查是否已有可信快取結果
- 若無，透過規劃器決定是否需要搜尋
- 使用搜尋工具取得外部資料
- 統整來源後產生最終查證回答
- 並將結果寫回快取供後續使用

---

## 🧠 系統流程說明（對應作業流程圖）

整體流程如下：

1. **input**
2. **check_cache**
   - 若快取命中（Cache Hit），直接回傳結果
   - 若未命中，進入 Planner
3. **planner**
   - 評估問題是否需要外部搜尋
   - 決定後續執行路徑
4. **query_gen**
   - 產生適合搜尋引擎使用的關鍵字
5. **search_tool**
   - 進行外部搜尋（如 searxNG）
6. **final_answer**
   - 整理證據、產生最終查證結果
7. **end**

---

## ⚠️ 關於流程圖「路徑多一條」的說明（重要補充）

在實際執行時，透過 `app.get_graph()` 或 Mermaid / ASCII 輸出所顯示的流程圖中，  
**planner 節點看起來會有「多一條連線」的情況**，例如：

- planner → search_tool（conditional）
- planner → query_gen（conditional）
- planner → final_answer（conditional）

視覺上可能會誤以為有「重複邊」或「多餘路徑」。

### ✅ 實際狀況說明

- **這不是程式邏輯錯誤**
- 在實際 Graph 結構中：
  - 每個 conditional edge 只定義一次
  - 同一時間只會走其中一條路徑
- 顯示上的「多一條線」屬於 **LangGraph 圖形視覺化的呈現問題**
  - 多個 conditional edge 疊加後，視覺上會被畫成多條線
  - 但實際執行時不會同時觸發

### 🔍 驗證方式

透過實際列印 Graph Edges 可確認：

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
