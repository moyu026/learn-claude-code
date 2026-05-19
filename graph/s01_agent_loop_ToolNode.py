import os
import subprocess
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


# ============================================================
# 1. 定义 AgentState
# ============================================================
# 这里和你原来的版本不同：
#
# 原来是：
#   messages: list[BaseMessage]
#
# 现在改成：
#   messages: Annotated[list[BaseMessage], add_messages]
#
# add_messages 是 LangGraph 的 reducer。
#
# 它的作用是：
# 当某个节点返回：
#   {"messages": [new_message]}
#
# LangGraph 不会覆盖原来的 messages，
# 而是自动把 new_message 追加进去。
#
# 这和 ToolNode 很适配。
#
# 因为 ToolNode 默认返回的是：
#   {"messages": [ToolMessage(...), ToolMessage(...)]}
#
# 如果没有 add_messages，messages 可能会被覆盖。
#
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 2. 定义 bash 工具
# ============================================================
@tool("bash")
def run_bash(command: str) -> str:
    """Run a shell command."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


TOOLS = [run_bash]


# ============================================================
# 3. 绑定工具到 LLM
# ============================================================
LLM = ChatOpenAI(
    model=MODEL,
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    max_tokens=8000,
).bind_tools(TOOLS)


# ============================================================
# 4. LLM 节点
# ============================================================
# 因为 messages 已经使用 add_messages reducer，
# 所以这里不需要手动写：
#
#   state["messages"] + [response]
#
# 只需要返回：
#
#   {"messages": [response]}
#
# LangGraph 会自动追加。
#
def call_model(state: AgentState) -> AgentState:
    response = LLM.invoke(
        [
            ("system", SYSTEM),
            *state["messages"],
        ]
    )

    return {
        "messages": [response],
    }


# ============================================================
# 5. 路由函数
# ============================================================
# 这个函数还是保留。
#
# 它决定 LLM 节点之后：
#
# - 如果模型产生了 tool_calls，就进入 tools 节点
# - 如果没有 tool_calls，就结束
#
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"


# ============================================================
# 6. 构建 Graph
# ============================================================
def build_graph():
    graph = StateGraph(AgentState)

    # LLM 节点
    graph.add_node("llm", call_model)

    # ========================================================
    # 关键变化：
    #
    # 原来是：
    #   graph.add_node("tools", execute_tools)
    #
    # 现在是：
    #   graph.add_node("tools", ToolNode(TOOLS))
    #
    # ToolNode 会自动完成：
    #   1. 读取最后一条 AIMessage.tool_calls
    #   2. 根据 tool name 找到对应工具
    #   3. 执行工具
    #   4. 生成 ToolMessage
    #   5. 返回 {"messages": [ToolMessage(...)]}
    #
    # ========================================================
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # 工具执行完成后，把 ToolMessage 交回 LLM。
    graph.add_edge("tools", "llm")

    return graph.compile()


AGENT_GRAPH = build_graph()


# ============================================================
# 7. Agent 执行函数
# ============================================================
def agent_loop(messages: list):
    final_state = AGENT_GRAPH.invoke(
        {"messages": messages},
        config={"recursion_limit": 100},
    )

    messages[:] = final_state["messages"]


# ============================================================
# 8. CLI 入口
# ============================================================
if __name__ == "__main__":
    history = []

    while True:
        try:
            query = input("\033[36mgraph-s01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append(HumanMessage(content=query))

        agent_loop(history)

        response = history[-1]

        if isinstance(response, AIMessage):
            print(response.content)

        print()
