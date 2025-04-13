from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

translation_prompt = PromptTemplate.from_template(
    "次の文章を{language}に翻訳し、翻訳された文章だけ答えてください。\n```\n{text}\n\```"
)

llm = ChatDeepSeek(model="deepseek-chat")
parser = StrOutputParser()

# LLM出力を表示するための関数
def display_llm_output(input_data):
    print(f"LLM出力: {input_data.content}")
    return input_data

# 表示機能付きLLM
llm_with_display = llm.with_fallbacks([]).with_config({"callbacks": []}) | display_llm_output

translation = translation_prompt | llm_with_display | parser

to_english = {
    "text": RunnablePassthrough(),
    "language": lambda _: "English"
} | translation

to_japanese = {
    "text": RunnablePassthrough(),
    "language": lambda _: "Japanese"
} | translation

chain = to_english | llm_with_display | parser | to_japanese

text = input("User: ")

print("処理を開始します...")
answer = chain.invoke({"text": text})

print(f"AI: {answer}")
