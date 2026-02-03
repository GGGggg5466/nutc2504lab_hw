from openai import OpenAI

# 你的 vLLM server（OpenAI-compatible）
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm-token",  # vLLM 通常不驗證，留著即可
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # ✅ 改成你 /v1/models 看到的 id

SYSTEM_PROMPT = "你是一個繁體中文的聊天機器人，請簡潔回答。"

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    print("AI:", response.choices[0].message.content)
