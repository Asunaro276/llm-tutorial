from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from typing import Literal

angle = 50

@tool
def get_current_light_angle() -> float:
    """
    現在の照明角度を取得します。
    """
    return angle

@tool
def set_light_angle(direction: Literal["right", "left"]) -> bool:
    """
    照明角度を設定します。
    成功した場合はTrueを返します。
    回転後の角度は get_current_light_angle で取得できます。
    """
    global angle
    angle += 10 if direction == "right" else -10
    angle += 360 if angle < 0 else 0
    return True

tools = [get_current_light_angle, set_light_angle]

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        """
        あなたの名前はハルです。
        あなたの仕事は宇宙船の制御です。
        ツール呼び出し毎に計器を確認してください。
        必ずCoT推論を行ってからツール呼び出しを行ってください。
        推論の過程も必ず示してください。
        """
    ),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatDeepSeek(model="deepseek-chat")
# llm = ChatOpenAI(model="gpt-4o")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
answer = agent_executor.invoke({"input": "ハル、ポッドのライトを20度左に回してくれ。", "chat_history": []})
print("回答: ", answer["output"])
