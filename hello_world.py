from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="qwen2.5-coder-7b-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    base_url="http://127.0.0.1:1234/v1",
)
res = llm.invoke("Hello, world!")
print(res)