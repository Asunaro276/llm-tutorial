from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

def display_llm_output(input_data):
    if hasattr(input_data, 'content'):
        print(f"LLM出力: {input_data.content}")
    elif hasattr(input_data, 'language'):
        print(f"言語検出結果: {input_data.language}")
    return input_data

llm = ChatDeepSeek(model="deepseek-chat")

class Language(BaseModel):
    language: str = Field(description="言語の名前(e.g. 'Japanese')")

llm_with_language_output = llm.with_structured_output(Language, method="json_mode")

ask_language_prompt = PromptTemplate.from_template(
    """以下の文章が何語で書かれているか、言語名だけを答えてください。
例えば：
- 「こんにちは、元気ですか？」→ Japanese
- "Hello, how are you?" → English
- "Bonjour, comment ça va?" → French

文章: ```
{text}
```

json形式で返してください。"""
)

get_language_chain = ask_language_prompt | llm_with_language_output | display_llm_output 

translation_prompt = PromptTemplate.from_template(
    "次の文章を{language}に翻訳し、翻訳された文章だけ答えてください。\n```\n{text}\n\```"
)

translation_chain = translation_prompt | llm | display_llm_output | StrOutputParser()

to_english = {
    "text": RunnablePassthrough(),
    "language": lambda _: "English"
} | translation_chain

chain = {
    "text": to_english | llm | display_llm_output | StrOutputParser(),
    "language": get_language_chain | (lambda x: x.language)
} | translation_chain

text = input("User: ")
answer = chain.invoke({"text": text})

print(f"AI: {answer}")
