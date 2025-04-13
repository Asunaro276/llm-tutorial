import functools
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
import operator

salesman_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
    あなたは非常に有益なWiFi回線を売っている熱意あるベテラン訪問販売員、田中太郎です。
    このWi-Fi回線はオトクな乗り換えキャンペーンをやっており、かなりの確率で契約を結ぶことができるほど魅力的な商品です。
    営業が終了したら、「FINISH」と回答してください。
    Wi-Fi契約がされるか、売れる見込みがない場合に営業を終了してください。
    """),
    HumanMessage(content="""
    こんにちは。どちら様でしょうか？
    """),
    ("placeholder", "{messages}"),
])

shed_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
    あなたは堅実な主夫の堅木実です。
    基本的に訪問販売の類はすべて断っている一方で、現在のWi-Fi回線が不安定なことが少し気になっていて、インターネット等でも頻繁に乗り換えのための検索をしています。
    """),
    ("placeholder", "{messages}"),
])

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

salesman_agent = salesman_prompt | llm
shed_agent = shed_prompt | llm

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def agent_node(state: AgentState, agent, name):
    result = agent.invoke(state)
    message = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    print(f"{name}: {message.content}")
    return {"messages": [message]}

def route(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if "FINISH" in last_message.content:
        return "finish"
    return "continue"

salesman_node = functools.partial(agent_node, agent=salesman_agent, name="Salesman")
shed_node = functools.partial(agent_node, agent=shed_agent, name="Shed")

workflow = StateGraph(AgentState)
workflow.add_node("Salesman", salesman_node)
workflow.add_node("Shed", shed_node)
workflow.add_conditional_edges(
    "Salesman",
    route,
    {"continue": "Shed", "finish": END}
)
workflow.add_edge("Shed", "Salesman")
workflow.set_entry_point("Salesman")
graph = workflow.compile()
graph.invoke({"messages": []})
