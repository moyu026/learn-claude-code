#!/usr/bin/env python3
# 示例目标：用 LangGraph 表达带上下文隔离的 subagent。
"""
s04_subagent_graph.py - 使用 LangGraph 改写 Subagents 示例

这是 agents/s04_subagent.py 的 LangGraph / langchain_openai 版本。

原版 s04 的核心思想是：

- 父 agent 拥有自己的 messages
- 父 agent 可以调用 `task` 工具，把一个子任务交给子 agent
- 子 agent 从空白上下文开始，只共享文件系统，不共享父 agent 的对话历史
- 子 agent 完成后，只把最终摘要返回给父 agent
- 子 agent 的中间工具调用和长上下文都会被丢弃

也就是：

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- 全新上下文
    |                  |  task       |                  |
    | tool: task       | ----------> | LLM -> tools     |
    | prompt="..."     |             | tools -> LLM     |
    |                  |  summary    |                  |
    | result="..."     | <---------- | 只返回最终文本    |
    +------------------+             +------------------+

核心思想：执行隔离天然带来上下文隔离。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

# LangChain 消息类型。
#
# AIMessage：
#   模型回复。模型请求工具调用时，tool_calls 字段里会有工具名、参数和 id。
#
# BaseMessage：
#   所有消息类型的基类，用于给 LangGraph state 做类型标注。
#
# HumanMessage：
#   用户消息。父 agent 的用户输入、子 agent 的任务 prompt 都用它表示。
#
# SystemMessage：
#   系统提示词。父 agent 和子 agent 使用不同的 system prompt。
#
# ToolMessage：
#   工具执行结果。它必须带 tool_call_id，才能和 AIMessage.tool_calls 对上。
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables.graph import MermaidDrawMethod

# @tool 把普通 Python 函数包装成 LangChain Tool。
# 包装后可以交给 ChatOpenAI.bind_tools(...) 生成 OpenAI-compatible 工具 schema。
from langchain_core.tools import tool

# ChatOpenAI 是 LangChain 的 OpenAI-compatible 模型入口。
# graph 目录下的示例统一使用它，而不是 Anthropic SDK。
from langchain_openai import ChatOpenAI

# StateGraph 用来定义有状态图；START 是图的起点。
from langgraph.graph import START, StateGraph

# add_messages 让节点返回的新消息追加到历史里，而不是覆盖原 messages。
from langgraph.graph.message import add_messages

# tools_condition 是预置路由函数：
# - 最后一条 AIMessage 有 tool_calls -> "tools"
# - 没有 tool_calls -> "__end__"
from langgraph.prebuilt import tools_condition

# 父 agent 的 CLI 多轮对话需要记忆；MemorySaver 是内存版 checkpoint。
from langgraph.checkpoint.memory import MemorySaver


# ------------------------------------------------------------
# 1. 环境变量和模型配置
# ------------------------------------------------------------

load_dotenv(override=True)

# 所有 shell 命令和文件工具都限制在当前工作目录。
WORKDIR = Path.cwd()

# graph 版本使用 ChatOpenAI，因此优先使用 OPENAI_MODEL。
#
# .env.example 里 MODEL_ID 是给 agents/*.py 的 Anthropic SDK 用的；
# OPENAI_MODEL 才是给 graph/*.py 的 OpenAI-compatible 调用用的。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# 父 agent 的系统提示词：
# 重点是告诉模型可以使用 task 工具做探索或拆分子任务。
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use the task tool to delegate exploration or subtasks."
)

# 子 agent 的系统提示词：
# 子 agent 只负责完成传入的任务，然后总结结果。
SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the given task, then summarize your findings."
)


# ------------------------------------------------------------
# 2. 路径安全检查
# ------------------------------------------------------------

def safe_path(p: str) -> Path:
    """
    把模型传入的路径转换为工作区内的绝对路径。

    允许：
        "README.md"
        "src/main.py"

    禁止：
        "../secret.txt"
        "/etc/passwd"

    父 agent 和子 agent 共享同一个文件系统，因此两边的文件工具都必须
    使用同一套路径边界检查。
    """
    path = (WORKDIR / p).resolve()

    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")

    return path


# ------------------------------------------------------------
# 3. 基础工具：父 agent 和子 agent 共享
# ------------------------------------------------------------

@tool("bash")
def run_bash(command: str) -> str:
    """
    在当前工作区执行 shell 命令。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
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


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """
    读取当前工作区内的文件。

    参数：
    - path：相对于工作区的路径
    - limit：可选，只返回前 N 行
    """
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """
    写入当前工作区内的文件。

    父目录不存在时会自动创建；文件已存在时会整体覆盖。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"

    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    在工作区文件中替换第一次出现的精确文本。
    """
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")

        if old_text not in content:
            return f"Error: Text not found in {path}"

        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


# 子 agent 只能使用基础工具。
# 它不能再调用 task，否则会出现递归创建 subagent 的复杂情况。
CHILD_TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
]


# ------------------------------------------------------------
# 4. LangGraph State
# ------------------------------------------------------------

class AgentState(TypedDict):
    """
    父图和子图共用的状态结构。

    messages 使用 add_messages reducer：
    节点返回 {"messages": [new_message]} 时，新消息会追加到历史后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------
# 5. 模型初始化
# ------------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)

# 子 agent 绑定基础工具。
child_llm = llm.bind_tools(CHILD_TOOLS)


# ------------------------------------------------------------
# 6. 通用工具执行函数
# ------------------------------------------------------------

def execute_tool_calls(state: AgentState, tool_by_name: dict[str, object], prefix: str) -> dict:
    """
    执行最后一条 AIMessage 中的所有 tool_calls。

    参数：
    - state：当前 LangGraph 状态
    - tool_by_name：工具名到 LangChain Tool 对象的映射
    - prefix：打印 trace 时使用的前缀，用来区分 parent / child

    返回：
    - {"messages": [ToolMessage(...), ...]}

    这里显式生成 ToolMessage，而不是依赖 Anthropic 的 tool_result 消息块。
    这是 OpenAI-compatible tool calling 在 LangChain 里的标准消息协议。
    """
    last = state["messages"][-1]

    if not isinstance(last, AIMessage):
        return {}

    tool_messages = []

    for call in last.tool_calls or []:
        tool_name = call["name"]
        tool_args = call.get("args", {})
        tool_obj = tool_by_name.get(tool_name)

        if tool_obj is None:
            output = f"Unknown tool: {tool_name}"
        else:
            try:
                output = tool_obj.invoke(tool_args)
            except Exception as e:
                output = f"Error: {e}"

        print(f"> {prefix}{tool_name}:")
        print(str(output)[:200])

        tool_messages.append(
            ToolMessage(
                content=str(output)[:50000],
                tool_call_id=call["id"],
            )
        )

    return {"messages": tool_messages}


def message_text(message: BaseMessage) -> str:
    """
    把 LangChain message.content 转成普通字符串。

    大多数 OpenAI-compatible 模型会返回字符串 content；少数适配器可能返回
    list[dict] 形式的结构化内容，因此这里做一个小兼容。
    """
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


# ------------------------------------------------------------
# 7. 子 agent 图：全新上下文，只返回最终摘要
# ------------------------------------------------------------

CHILD_TOOL_BY_NAME = {tool.name: tool for tool in CHILD_TOOLS}


def child_agent_node(state: AgentState) -> dict:
    """
    子 agent 的模型节点。

    子 agent 每次调用模型时使用 SUBAGENT_SYSTEM。
    它的 messages 从 run_subagent(prompt) 创建，不包含父 agent 的历史。
    """
    response = child_llm.invoke(
        [SystemMessage(content=SUBAGENT_SYSTEM)] + state["messages"]
    )

    return {"messages": [response]}


def child_tools_node(state: AgentState) -> dict:
    """
    子 agent 的工具节点。

    子 agent 只能访问基础工具，不能访问 task 工具。
    """
    return execute_tool_calls(state, CHILD_TOOL_BY_NAME, prefix="child.")


child_builder = StateGraph(AgentState)
child_builder.add_edge(START, "agent")
child_builder.add_node("agent", child_agent_node)
child_builder.add_node("tools", child_tools_node)
child_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "__end__",
    },
)
child_builder.add_edge("tools", "agent")

# 子图不使用 checkpointer。
# 每次 run_subagent(...) 都从 {"messages": [HumanMessage(prompt)]} 开始，
# 因此天然是全新上下文。
child_graph = child_builder.compile()


def run_subagent(prompt: str) -> str:
    """
    启动一个子 agent。

    注意这里的隔离边界：
    - 子 agent 的初始 messages 只有 prompt
    - 子 agent 可以读写同一个 WORKDIR
    - 子 agent 的完整消息历史不会返回给父 agent
    - 父 agent 只收到最后一条 AIMessage 的文本摘要
    """
    final_state = child_graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": 30},
    )

    last = final_state["messages"][-1]

    if isinstance(last, AIMessage):
        text = message_text(last).strip()
        return text or "(no summary)"

    return "(no summary)"


# ------------------------------------------------------------
# 8. 父 agent 工具：基础工具 + task 调度器
# ------------------------------------------------------------

@tool("task")
def run_task(prompt: str, description: str | None = None) -> str:
    """
    启动一个拥有全新上下文的子 agent。

    子 agent 和父 agent 共享文件系统，但不共享对话历史。
    参数：
    - prompt：交给子 agent 的完整任务说明
    - description：可选，子任务的简短描述，用于终端 trace
    """
    desc = description or "subtask"
    print(f"> task ({desc}): {prompt[:80]}")
    return run_subagent(prompt)


PARENT_TOOLS = CHILD_TOOLS + [run_task]
PARENT_TOOL_BY_NAME = {tool.name: tool for tool in PARENT_TOOLS}

# 父 agent 绑定基础工具和 task 工具。
parent_llm = llm.bind_tools(PARENT_TOOLS)


# ------------------------------------------------------------
# 9. 父 agent 图：正常对话历史 + task 工具
# ------------------------------------------------------------

def parent_agent_node(state: AgentState) -> dict:
    """
    父 agent 的模型节点。

    父 agent 使用 SYSTEM，并且可以看到完整父会话历史。
    如果它需要隔离探索，可以调用 task 工具。
    """
    response = parent_llm.invoke(
        [SystemMessage(content=SYSTEM)] + state["messages"]
    )

    return {"messages": [response]}


def parent_tools_node(state: AgentState) -> dict:
    """
    父 agent 的工具节点。

    它可以执行基础工具，也可以执行 task。
    当 task 被执行时，会同步运行子图，并把子 agent 的最终摘要作为工具结果
    返回给父 agent。
    """
    return execute_tool_calls(state, PARENT_TOOL_BY_NAME, prefix="")


parent_builder = StateGraph(AgentState)
parent_builder.add_edge(START, "agent")
parent_builder.add_node("agent", parent_agent_node)
parent_builder.add_node("tools", parent_tools_node)
parent_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "__end__",
    },
)
parent_builder.add_edge("tools", "agent")

# 父图使用 MemorySaver，这样 CLI 中同一个 thread_id 可以连续多轮对话。
memory = MemorySaver()
graph = parent_builder.compile(checkpointer=memory)


# ------------------------------------------------------------
# 10. 图可视化辅助函数
# ------------------------------------------------------------

def save_graph_png(output_dir: str | Path = "graph") -> tuple[Path, Path]:
    """
    把父图和子图导出成 Mermaid PNG。

    LangGraph 编译后的 graph 对象支持：

        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )

    这里分别导出：
    - s04_parent_graph.png：父 agent 图，包含 task 工具
    - s04_child_graph.png：子 agent 图，不包含 task 工具

    MermaidDrawMethod.API 会调用 Mermaid 在线渲染 API；
    如果当前环境无法访问网络，这一步可能失败，但不影响 agent 正常运行。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_png = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    child_png = child_graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )

    parent_path = out_dir / "s04_parent_graph.png"
    child_path = out_dir / "s04_child_graph.png"

    parent_path.write_bytes(parent_png)
    child_path.write_bytes(child_png)

    return parent_path, child_path


# ------------------------------------------------------------
# 11. 对外调用封装
# ------------------------------------------------------------

def run_once(query: str, thread_id: str = "default") -> str:
    """
    执行一次父 agent 用户输入，并返回最终模型文本。

    同一个 thread_id 会复用父 agent 的消息历史。
    子 agent 不复用历史，每次 task 都是全新上下文。
    """
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
        },
    )

    return message_text(final_state["messages"][-1])


def agent_loop(messages: list[BaseMessage]) -> None:
    """
    兼容原版 s04 的 agent_loop(messages) 调用方式。

    外部可以自己维护 messages 列表；这里执行完成后会用最终 state 覆盖原列表。
    为避免 MemorySaver 对同一个列表重复追加，thread_id 使用对象 id 和当前长度生成。
    """
    final_state = graph.invoke(
        {"messages": messages},
        config={
            "configurable": {"thread_id": f"agent-loop-{id(messages)}-{len(messages)}"},
            "recursion_limit": 100,
        },
    )

    messages[:] = final_state["messages"]


# ------------------------------------------------------------
# 12. CLI 入口
# ------------------------------------------------------------

if __name__ == "__main__":
    if "--draw-graph" in sys.argv:
        arg_index = sys.argv.index("--draw-graph")
        output_dir = sys.argv[arg_index + 1] if len(sys.argv) > arg_index + 1 else "graph"
        parent_path, child_path = save_graph_png(output_dir)
        print(f"父图已保存：{parent_path}")
        print(f"子图已保存：{child_path}")
        raise SystemExit(0)

    if "--chat" not in sys.argv:
        parent_path, child_path = save_graph_png("graph")
        print(f"父图已保存：{parent_path}")
        print(f"子图已保存：{child_path}")
        print("如需进入交互式 agent，请运行：python graph/s04_subagent_graph.py --chat")
        raise SystemExit(0)

    thread_id = "cli-session"

    while True:
        try:
            query = input("\033[36mgraph-s04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        answer = run_once(query, thread_id=thread_id)
        print(answer)
        print()
