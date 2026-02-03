from openai import OpenAI

client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token"   # vLLM 沒驗證也可留著
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role":"user", "content":"用一句話介紹 vLLM 是什麼"}],
    max_tokens=80,
)

print(resp.choices[0].message.content)
