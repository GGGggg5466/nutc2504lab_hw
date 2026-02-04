# meeting_assistant 智慧會議記錄助手

## 結構
- input/：放音檔（例如 Podcast_EP14.wav）
- tools/：工具腳本（ASR）
  - HW-asr.py：語音轉文字（輸出 .txt / .srt）
- output/：本程式產出的結果
  - transcript.txt：純文字逐字稿
  - minutes.md：會議記錄（條列整理）
  - summary.md：重點摘要
  - report.md：報告式整理
- ch_meeting_graph.py：主程式（串 ASR + LLM + 輸出）


