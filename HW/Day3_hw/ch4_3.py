import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL    = os.getenv("OPENAI_MODEL")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0
)

# =========================
# ch4-3 Tool：科技文章摘要工具
# =========================
@tool
def generate_tech_summary(article_content: str) -> str:
    """
    科技文章專用摘要生成工具。

    【判斷邏輯】
    1. 只有當輸入內容屬於「科技」、「程式設計」、「AI」、「軟體工程」或「IT 技術」領域時，才使用此工具。
    2. 如果內容是「閒聊」、「食譜」、「天氣」、「日常日記」等非技術內容，請勿使用此工具。

    功能：將輸入的技術文章歸納出 3 個重點（Key Takeaways），使用繁體中文。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個資深的科技主編。請將輸入的技術文章內容，精簡地歸納出 3 個關鍵重點（Key Takeaways）。請用繁體中文輸出，條列清楚。"),
        ("user", "{text}")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": article_content})
    return result


# =========================
# ch4-3 Router：決定要不要用工具
# =========================
llm_with_tools = llm.bind_tools([generate_tech_summary])

router_prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

def main():
    print("=== ch4-3 interactive router ===")
    print("輸入一段文字：若是科技文章會自動摘要；若不是科技文章就直接回覆。")
    print("輸入 exit 或 q 離開。\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break

        chain = router_prompt | llm_with_tools
        ai_msg = chain.invoke({"input": user_input})

        tool_calls = getattr(ai_msg, "tool_calls", None)

        if tool_calls:
            print("✅ [決策] 判斷為科技文章")
            tool_args = tool_calls[0]["args"]

            final_result = generate_tech_summary.invoke(tool_args)

            print(f"📌 [執行結果]:\n{final_result}\n")
        else:
            print("❌ [決策] 判斷為閒聊/非科技文章，直接回答")
            print(f"💬 [AI 回應]: {ai_msg.content}\n")


if __name__ == "__main__":
    main()
