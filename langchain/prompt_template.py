import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), openai_api_base="https://api.deepseek.com")
prompt_template = PromptTemplate.from_template(
    "{Dish}を調理するための材料を教えてください。"
)
prompt = prompt_template.invoke({"Dish": "ハンバーグ"})
print(prompt)
print(llm.invoke(prompt))
