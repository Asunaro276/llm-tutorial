from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), openai_api_base="https://api.deepseek.com")
    response = llm.invoke("こんにちは")
    print(response)



if __name__ == "__main__":
    main()
