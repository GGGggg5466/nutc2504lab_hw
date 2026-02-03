import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
    model="google/gemma-3-27b-it",
    temperature=0.1,
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是資料提取助手。請從使用者輸入中擷取欄位：name, phone, product, quantity, address。"
     "只能輸出 JSON，不能有任何多餘文字。quantity 必須是數字。"),
    ("human", "{text}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

user_input = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週送到台中市北區。"

def clean_json(s: str) -> str:
    # 移除可能的 ```json code fence
    s = s.strip()
    s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return s

if __name__ == "__main__":
    raw = chain.invoke({"text": user_input})
    cleaned = clean_json(raw)
    try:
        obj = json.loads(cleaned)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception as e:
        print("❌ JSON 解析失敗")
        print("raw:\n", raw)
        print("cleaned:\n", cleaned)
        print("error:", e)
