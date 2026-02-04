import os
import json
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL    = os.getenv("OPENAI_MODEL")

@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """資料提取專用工具：從文字提取(姓名、電話、商品、數量、地址)"""
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address,
    }

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0
)

llm_with_tools = llm.bind_tools([extract_order_data])

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。若資訊不足或不是訂單，請用自然語言回答。"),
    ("user", "{user_input}")
])

# ✅ ch4-2 改良點：有 tool call -> 回傳 args（結構化）
#                 沒 tool call -> 回傳 content（非結構化）
def extract_tool_args(ai_message):
    tool_calls = getattr(ai_message, "tool_calls", None)
    if tool_calls:
        return tool_calls[0]["args"]
    else:
        return ai_message.content

chain = prompt | llm_with_tools | extract_tool_args

def pretty_print(result):
    """讓輸出一致漂亮：dict 以 JSON 印；str 直接印。"""
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 可能是 str 或其他型別
        print(str(result))

if __name__ == "__main__":
    print("=== ch4-2 interactive mode ===")
    print("輸入訂單句子讓我提取；輸入 exit 或 q 離開。\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break

        result = chain.invoke({"user_input": user_input})
        pretty_print(result)
        print()  
