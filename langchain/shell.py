from os import name
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
import operator
import subprocess

@tool
def exec_command(shell_command: str) -> str:
    """シェルコマンドを実行します
    shell_command: Linuxシェルコマンド
    """
    result = subprocess.run(shell_command, shell=True, capture_output=True)
    return result.stdout.decode("utf-8") + result.stderr.decode("utf-8")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tool = llm.bind_tools([exec_command])

def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    messages = []
    for call in last_message.tool_calls:
        if call["name"] == "exec_command":
            value = exec_command.invoke(call["args"])
            tool_message = ToolMessage(content=value, name=call["name"], tool_call_id=call["id"])
            messages.append(tool_message)
    return {"messages": messages}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool"
    else:
        return "end"

def main():
    workflow = StateGraph(AgentState)
    workflow.add_node("Agent", agent_node)
    workflow.add_node("Tool", tool_node)
    workflow.add_conditional_edges(
        "Agent",
        should_continue,
        {"tool": "Tool", "end": END}
    )
    workflow.add_edge("Tool", "Agent")
    workflow.set_entry_point("Agent")
    graph = workflow.compile()

    query = input("query: ")
    state = graph.invoke({"messages": [HumanMessage(content=query)]})
    print(state["messages"][-1].content)

if __name__ == "__main__":
    main()
