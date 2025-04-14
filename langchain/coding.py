import subprocess
import functools
import json
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing import Annotated, Sequence, Union, Literal
from langgraph.graph import StateGraph, END
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict
import operator
from dotenv import load_dotenv

load_dotenv()


@tool
def evaluate(code: str, test: str) -> tuple[str, str]:
    """
    コードをテストし、標準出力と標準エラーのペアを返す。
    code: テスト対象のコード
    test: pytestのテストコード
    """
    with open("product.py", "w") as f:
        f.write(code)
    with open("test_product.py", "w") as f:
        f.write(test)
    result = subprocess.run(["pytest", "test_product.py"], capture_output=True)
    return result.stdout.decode(), result.stderr.decode()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    task: str

class LeaderResponse(BaseModel):
    reasoning: str = Field(description="決定の背後にある理由")
    next: Union[Literal["Finish"], str] = Field(
        description = "次にアクションを行うチームメンバー"
    )
    instructions: str = Field(
        description = "次のチームメンバへの指示。"
    )

MEMBERS = {
    "Leader": (
        "チームの進行を管理し、チームメンバの役割を割り当てます"
        "チームの目標を達成した場合はFinishを選択します。\n"
    ),
    "Programmer": "仕様に基づいてコードを書きます。\n",
    "TestWriter": "仕様に基づいてテストを書きます。\n",
    "Evaluator": (
        "コードとテストを実行し、テスト結果を分析します。"
        "コードとテストを実行するにはevaluateツールを使用します。\n"
    )
}

def create_agent(llm, name: str) -> ChatPromptTemplate:
    """特定の役割のためのプロンプトテンプレートを作成する。"""
    members = ", ".join(MEMBERS.keys())
    member_roles = "\n".join(f"{member}: {role}" for member, role in MEMBERS.items())
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"""
            あなたはソフトウェア開発チームの一員の{name}です。
            チームの最終成果物は、仕様を満たすことを確認したテスト済みのコードです。
            コードはproduct.pyに保存されるものとします。
            テストはtest_product.pyに保存されるものとします。
            必ずステップバイステップで推論の過程を説明してから答えてください。
            チームメンバー: {members}\n\n
            各メンバーの役割: {member_roles}
            """
        ),
        HumanMessagePromptTemplate.from_template("task: {task}\n"),
        MessagesPlaceholder("messages")
    ]).partial(name=name, members=members, member_roles=member_roles)
    return prompt | llm

llm = ChatDeepSeek(model="deepseek-chat")
# llm = ChatOpenAI(model="gpt-4o")
leader_agent = create_agent(llm.with_structured_output(LeaderResponse), "Leader")
programmer_agent = create_agent(llm, "Programmer")
tester_agent = create_agent(llm, "TestWriter")
evaluator_agent = create_agent(llm.bind_tools([evaluate]), "Evaluator")

def leader_node(state: AgentState) -> dict:
    """リーダーエージェントのためのノード関数"""
    response = leader_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=response.instructions, name="Leader")],
        "next": response.next,
    }

def member_node(state: AgentState, agent, name: str) -> dict:
    """汎用エージェントのためのノード関数"""
    for _ in range(10):
        result = agent.invoke(state)
        has_errors = False
        if result.tool_calls:
            for call in result.tool_calls:
                name = call["name"]
                if name != "evaluate":
                    has_errors = True
                    print(f"無効なツール呼び出し: {name}")
                    break
                args = call["args"]
                try:
                    evaluate.args_schema.model_validate(args)
                except Exception as e:
                    print(f"バリデーション失敗: {str(e)}")
                    has_errors = True
        if not has_errors:
            break
    else:
        raise ValueError("無効なツール呼び出しが続いたため、終了します。")
    return {
        "messages": [
            AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
        ]
    }

programmer_node = functools.partial(member_node, agent=programmer_agent, name="Programmer")
tester_node = functools.partial(member_node, agent=tester_agent, name="TestWriter")
evaluator_node = functools.partial(member_node, agent=evaluator_agent, name="Evaluator")

def router(state: AgentState) -> str:
    """ワークフローの次のステップを決定するルータ関数"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    return "continue"

def main():
    # ワークフロー定義
    workflow = StateGraph(AgentState)
    workflow.add_node("Leader", leader_node)
    workflow.add_node("Evaluator", evaluator_node)
    workflow.add_node("Tool", ToolNode([evaluate]))
    workflow.add_node("Programmer", programmer_node)
    workflow.add_node("TestWriter", tester_node)
    workflow.add_conditional_edges(
        "Evaluator",
        router,
        {
            "continue": "Leader",
            "call_tool": "Tool",
        }
    )
    workflow.add_conditional_edges(
        "Leader",
        lambda x: x["next"],
        {
            "Programmer": "Programmer",
            "TestWriter": "TestWriter",
            "Evaluator": "Evaluator",
            "Finish": END,
        }
    )
    workflow.add_edge("Programmer", "Leader")
    workflow.add_edge("TestWriter", "Leader")
    workflow.add_edge("Tool", "Evaluator")
    workflow.set_entry_point("Leader")

    graph = workflow.compile()
    with open("langchain/HumanEval.jsonl") as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines]
    entry = entries[0]

    result = graph.invoke({"messages": [], "task": entry["prompt"]})
    for message in result["messages"]:
        print(f"{message.name}: {message.content}")

if __name__ == "__main__":
    main()
