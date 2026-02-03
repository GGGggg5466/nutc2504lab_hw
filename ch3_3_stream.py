from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
    model="google/gemma-3-27b-it",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是專業科技文章編輯，請把內容整理成 3 個重點，用繁體中文條列。"),
    ("human", "{article_content}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

tech_article = """
大型語言模型（LLM）能理解並生成自然語言，常用於客服、摘要、翻譯與程式輔助。
在實務上，開發者需要處理提示詞設計、資料檢索、輸出格式控制與串流回應等問題。
"""

if __name__ == "__main__":
    print("=== 開始生成摘要（串流模式）===\n")
    for chunk in chain.stream({"article_content": tech_article}):
        print(chunk, end="", flush=True)
    print("\n\n=== Done ===")
