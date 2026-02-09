import pandas as pd

CSV_PATH = "s1411232035_RAG_HW_01.csv"

df = pd.read_csv(CSV_PATH)

# 確認資料結構
print(df[["q_id", "method", "score"]].head())

print("\n=== 各切塊方法的平均語意分數（20 題） ===")

avg_scores = (
    df.groupby("method")["score"]
      .mean()
      .sort_values(ascending=False)
)

for method, score in avg_scores.items():
    print(f"{method:10s} avg score = {score:.4f}")
