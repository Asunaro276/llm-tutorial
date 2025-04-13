from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

def main():
    llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), openai_api_base="https://api.deepseek.com")
    messages = [
        SystemMessage(content="あなたの名前はハルです。"),
        HumanMessage(content="私の名前はデイブです。"),
        AIMessage(content="こんにちは、デイブさん。"),
        HumanMessage(content="ハル、ポッドのライトを20度左に回してくれ。"),
    ]
    
    @tool
    def light_control(light_name: str, degree: int):
        """
        ライトを右にdegrees度回します。
        """
        # ここでライトを右に degreees 度回すコードを実装する
        print(f"{light_name}のライトを{degree}度回します。")
        return True

    llm_with_tool = llm.bind_tools([light_control])
    response = llm_with_tool.invoke(messages)
    messages.append(response)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            value = light_control.invoke(tool_call["args"])
            messages.append(ToolMessage(content=value, tool_call_id=tool_call["id"]))
    response = llm_with_tool.invoke(messages)
    print(response)



if __name__ == "__main__":
    main()
