import os, json
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
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。請用工具回傳結果。"),
    ("user", "{user_input}")
])

def extract_tool_args(ai_message):
    tool_calls = getattr(ai_message, "tool_calls", None)
    if tool_calls:
        return tool_calls[0]["args"]
    return None

chain = prompt | llm_with_tools | extract_tool_args

if __name__ == "__main__":
    user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"
    result = chain.invoke({"user_input": user_text})

    if result:
        print("✅ 提取成功：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("❌ 模型沒有發出 tool call（tool_calls 是空的）")
        msg = (prompt | llm_with_tools).invoke({"user_input": user_text})
        print("raw tool_calls:", getattr(msg, "tool_calls", None))
