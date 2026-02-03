from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) 連到你的 vLLM OpenAI-Compatible Server
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",  # vLLM 通常不驗證，填什麼都行
    model="google/gemma-3-27b-it",  # 先用你目前跑得動的
    temperature=0.7,
)

# 2) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是專業科技文章編輯，請把使用者提供的文章內容整理成 3 個重點，用繁體中文條列。"),
    ("human", "{article_content}")
])

# 3) Parser（把模型輸出當成純文字）
parser = StrOutputParser()

# 4) LCEL：Prompt -> LLM -> Parser
chain = prompt | llm | parser

tech_article = """
LangChain 是一個開源框架，用來協助開發者更容易把 LLM 串接到應用中，
它提供 prompt 模板、記憶、工具呼叫、代理等能力，並可結合資料庫與文件。
"""

if __name__ == "__main__":
    print("=== 開始生成摘要 ===")
    result = chain.invoke({"article_content": tech_article})
    print(result)
