#!/usr/bin/env python3
# Harness: the agent loop expressed as a LangGraph StateGraph.
r"""
s01_agent_loop_langgraph.py - LangGraph Agent Loop

This keeps the original s01 behavior, but moves the control flow into
LangGraph:

    +-------+    tool_use     +---------+
    |  llm  | --------------> |  tools  |
    +---+---+                 +----+----+
        ^                          |
        |       tool_result        |
        +--------------------------+

The graph stops when the model response has no tool calls.
"""

import os
import subprocess
from typing import TypedDict

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


class AgentState(TypedDict):
    messages: list[BaseMessage]


@tool("bash")
def run_bash(command: str) -> str:
    """Run a shell command."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


TOOLS = [run_bash]
LLM = ChatOpenAI(
    model=MODEL,
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    max_tokens=8000,
).bind_tools(TOOLS)


def call_model(state: AgentState) -> AgentState:
    response = LLM.invoke([("system", SYSTEM), *state["messages"]])
    return {
        "messages": state["messages"] + [response],
    }


def execute_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return state

    results = []
    for call in last_message.tool_calls:
        if call["name"] == "bash":
            command = call["args"]["command"]
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash.invoke({"command": command})
            print(output[:200])
        else:
            output = f"Unknown tool: {call['name']}"
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {
        "messages": state["messages"] + results,
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_model)
    graph.add_node("tools", execute_tools)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")
    return graph.compile()


AGENT_GRAPH = build_graph()


def agent_loop(messages: list):
    final_state = AGENT_GRAPH.invoke(
        {"messages": messages},
        config={"recursion_limit": 100},
    )
    messages[:] = final_state["messages"]


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
