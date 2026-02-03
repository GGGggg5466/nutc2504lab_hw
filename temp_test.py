from openai import OpenAI

client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
)

MODEL_ID = "google/gemma-3-27b-it"  # ✅ 用你 /v1/models 看到的 id

prompt = "請用100字形容『人工智慧』。"
temps = [0.1, 1.5]  # 0.1 很穩、1.5 很發散

for t in temps:
    print(f"\n➡️ 測試 Temperature = {t} ...")
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=t,
            max_tokens=200,   # 100字中文大概需要 150~250 tokens，保險一點
        )
        print("🤖 回覆：", resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
