from openai import OpenAI

# 連到你本機 vLLM OpenAI-compatible server
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token"   # vLLM 通常不驗證，隨便填也可
)

MODEL_ID = "google/gemma-3-27b-it"  # ✅要跟你 /v1/models 看到的一樣

history = [
    {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔回答。"}
]

print("輸入 exit 或 q 離開。")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ("exit", "q"):
        print("Bye!")
        break

    # 1) 把使用者輸入塞進歷史
    history.append({"role": "user", "content": user_input})

    try:
        # 2) 把整段 history 丟給模型
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=history,
            temperature=0.7,
            max_tokens=256,
        )

        assistant_reply = resp.choices[0].message.content
        print(f"AI: {assistant_reply}\n")

        # 3) 把模型回覆也塞回 history（多輪的關鍵）
        history.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        print(f"Error: {e}")
        # 出錯時可選擇把剛
