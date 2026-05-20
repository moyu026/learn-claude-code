#!/usr/bin/env python3
# 示例目标：用 LangGraph 表达带 TodoWrite 的 agent loop。
"""
s03_todo_write_graph.py - 使用 LangGraph 改写 TodoWrite 示例

这是 agents/s03_todo_write.py 的 LangGraph / langchain_openai 版本。

原版 Anthropic 写法是手动维护一个循环：

    LLM -> tool_use -> 执行工具 -> tool_result -> LLM -> ...

这里把同样的控制流改写成 LangGraph 的 StateGraph：

    +----------+      +---------+      +---------+
    |  START   | ---> |  agent  | ---> |  tools  |
    +----------+      +----+----+      +----+----+
                          |                |
                          | 无 tool_calls  |
                          v                |
                         END <-------------+

s03 相比 s02 的关键新增点是 TodoWrite：

- 模型可以调用 `todo` 工具维护结构化任务列表
- 同一时间只允许一个任务处于 `in_progress`
- 如果模型连续多轮使用工具但没有更新 todo，图会在下一次模型调用前插入提醒

核心思想：agent 自己维护进度，外部也能看到它当前在做什么。
"""

import os
import subprocess
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

# LangChain 的消息类型。
#
# BaseMessage：
#   所有聊天消息的基类。
#
# HumanMessage：
#   用户侧消息。这里除了真实用户输入，也会用它插入 todo 提醒。
#
# SystemMessage：
#   系统提示词。每次调用模型时临时放到 messages 最前面。
#
# AIMessage：
#   模型返回的消息。如果模型要调用工具，tool_calls 字段里会包含工具名、
#   调用 id 和参数。
#
# ToolMessage：
#   工具执行结果。OpenAI / LangChain 的工具调用协议要求：
#   AIMessage(tool_calls=[...]) 后面必须跟对应 tool_call_id 的 ToolMessage。
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# @tool 会把普通 Python 函数包装成 LangChain Tool。
#
# 包装后，函数名、类型标注和 docstring 会转换成工具 schema，
# 再通过 ChatOpenAI.bind_tools(...) 暴露给模型。
from langchain_core.tools import tool

# ChatOpenAI 是 LangChain 对 OpenAI-compatible chat model 的封装。
#
# 如果使用 OpenAI 官方接口，可以只配置 OPENAI_API_KEY。
# 如果使用兼容 OpenAI 协议的网关、本地服务或代理，则配置 OPENAI_BASE_URL
# 或 LLM_BASE_URL。
from langchain_openai import ChatOpenAI

# StateGraph 用来定义有状态工作流。
#
# START 是图的起点标记。
from langgraph.graph import START, StateGraph

# add_messages 是 LangGraph 的 reducer。
#
# 当某个节点返回 {"messages": [new_message]} 时，
# 它不会覆盖原来的 messages，而是把新消息追加到历史后面。
from langgraph.graph.message import add_messages

# tools_condition 是 LangGraph 预置的路由函数。
#
# 它会检查最后一条 AIMessage：
# - 如果有 tool_calls，就路由到 "tools"
# - 如果没有 tool_calls，就路由到 "__end__"
from langgraph.prebuilt import tools_condition

# MemorySaver 是内存版 checkpoint。
#
# CLI 多轮输入时，只要 thread_id 相同，就能复用上一轮的消息历史。
from langgraph.checkpoint.memory import MemorySaver


# ------------------------------------------------------------
# 1. 环境变量和模型配置
# ------------------------------------------------------------

load_dotenv(override=True)

# 当前工作目录。
# bash、read_file、write_file、edit_file 都会以这个目录为边界。
WORKDIR = Path.cwd()

# graph 目录下的版本使用 langchain_openai.ChatOpenAI，因此优先读取 OPENAI_MODEL。
#
# .env.example 里同时包含两套配置：
# - MODEL_ID：给 agents/*.py 的 Anthropic client 使用
# - OPENAI_MODEL：给 graph/*.py 的 OpenAI-compatible client 使用
#
# 如果这里优先读取 MODEL_ID，就会把 claude-sonnet-4-6 之类的 Anthropic
# 模型名传给 OpenAI-compatible 网关，常见结果是 model not found / invalid model。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# 系统提示词保持短而明确。
#
# 它告诉模型：
# - 当前工作目录在哪里
# - 多步骤任务要用 todo 规划
# - 开始某个任务前标记 in_progress，完成后标记 completed
# - 优先通过工具行动，而不是只输出解释
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# ------------------------------------------------------------
# 2. TodoManager：由模型写入的结构化进度状态
# ------------------------------------------------------------

class TodoManager:
    """
    一个简单的内存 todo 管理器。

    模型不能直接修改 Python 对象，而是通过 `todo` 工具提交完整任务列表。
    TodoManager 负责校验列表是否合法，并保存最新状态。

    这里保留原版 s03 的规则：
    - 最多 20 个 todo
    - 每个 todo 必须有 id / text / status
    - status 只能是 pending、in_progress、completed
    - 同一时间最多只能有一个 in_progress
    """

    def __init__(self):
        self.items: list[dict[str, str]] = []

    def update(self, items: list[dict]) -> str:
        """
        校验并替换当前 todo 列表。

        这里要求模型每次传入完整列表，而不是只传增量变化。
        好处是状态更容易观察和复现：工具返回的渲染结果就是当前完整进度。
        """
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            if not text:
                raise ValueError(f"Item {item_id}: text required")

            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")

            if status == "in_progress":
                in_progress_count += 1

            validated.append(
                {
                    "id": item_id,
                    "text": text,
                    "status": status,
                }
            )

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        把 todo 列表渲染成终端里容易读的格式。

        标记沿用原版 s03：
        - [ ] 表示 pending
        - [>] 表示 in_progress
        - [x] 表示 completed
        """
        if not self.items:
            return "No todos."

        lines = []

        for item in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")

        return "\n".join(lines)


# 和原版脚本一样，这里使用一个进程级 TODO 对象。
#
# 如果做成真正的多用户服务，应该把 todo 状态按 thread_id 或用户隔离，
# 不要使用全局变量共享所有会话。
TODO = TodoManager()


# ------------------------------------------------------------
# 3. 文件路径安全检查
# ------------------------------------------------------------

def safe_path(p: str) -> Path:
    """
    把模型传入的路径转换成工作区内的绝对路径。

    允许：
        "a.txt"
        "src/main.py"

    禁止：
        "../secret.txt"
        "/etc/passwd"

    这个检查很重要，因为模型可以调用 write_file / edit_file。
    如果不限制路径，模型的一次错误工具调用可能会改到工作区之外。
    """
    path = (WORKDIR / p).resolve()

    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")

    return path


# ------------------------------------------------------------
# 4. 工具实现
# ------------------------------------------------------------
#
# 原版 Anthropic 写法是：
#
#   TOOLS = [{"name": "...", "input_schema": ...}]
#   TOOL_HANDLERS = {"bash": lambda **kw: ...}
#
# LangChain / LangGraph 更常见的写法是：
#
#   @tool("name")
#   def tool_func(...):
#       """给模型看的工具描述"""
#
# 然后把工具对象交给 llm.bind_tools(TOOLS)，告诉模型有哪些工具、
# 参数 schema 是什么。
#
# 本文件的工具执行逻辑放在 tools_node 里显式完成。这样可以在执行工具后
# 插入 s03 特有的 todo reminder 逻辑。

@tool("bash")
def run_bash(command: str) -> str:
    """
    在当前工作区执行 shell 命令。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # shell=True 方便模型生成复合命令。
        # cwd=WORKDIR 把命令执行位置限制在当前项目。
        # capture_output=True 捕获 stdout / stderr，作为工具结果返回给模型。
        # timeout=120 防止命令长期挂起。
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()

        # 工具结果会进入模型上下文，所以要截断，避免一次命令输出撑爆上下文。
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

    except Exception as e:
        return f"Error: {e}"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """
    读取当前工作区内的文件内容。

    参数：
    - path：相对于工作区的文件路径
    - limit：可选，只返回前 N 行
    """
    try:
        fp = safe_path(path)
        lines = fp.read_text(encoding="utf-8").splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """
    写入当前工作区内的文件。

    如果父目录不存在，会自动创建；如果文件已存在，会整体覆盖。
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


@tool("todo")
def run_todo(items: list[dict]) -> str:
    """
    更新任务列表，用于跟踪多步骤任务进度。

    每个 item 必须包含：
    - id：稳定的字符串 id
    - text：任务描述
    - status：pending、in_progress 或 completed
    """
    return TODO.update(items)


# 工具定义的唯一来源。
# 这份列表会提供给 bind_tools。
TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
    run_todo,
]

# 执行工具时需要按名字查找 LangChain Tool 对象。
# @tool 包装后的对象会有 name 属性，这个名字就是模型 tool_calls 里的 name。
TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}


# ------------------------------------------------------------
# 5. LangGraph State
# ------------------------------------------------------------

class AgentState(TypedDict):
    """
    LangGraph 节点之间共享的状态。

    messages：
        完整对话历史。因为使用了 add_messages，节点返回新消息时会追加到历史，
        不会覆盖整个列表。

    rounds_since_todo：
        记录连续多少轮工具执行没有调用 todo。
        当这个值达到 3 时，tools_node 会插入：

            <reminder>Update your todos.</reminder>

        这就是原版 s03 里 nag reminder 的 LangGraph 写法。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    rounds_since_todo: int


# ------------------------------------------------------------
# 6. 初始化模型
# ------------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)

# bind_tools 只负责把工具 schema 暴露给模型。
#
# 它不会执行工具；真正执行工具的是后面的 tools_node。
llm_with_tools = llm.bind_tools(TOOLS)


# ------------------------------------------------------------
# 7. Graph 节点
# ------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    调用模型一次。

    输入：
        state["messages"] 是当前对话历史。

    输出：
        {"messages": [response]}

    因为 messages 使用 add_messages reducer，response 会被追加到历史后面。
    """
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM)] + state["messages"]
    )

    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行最后一条 AIMessage 中的所有工具调用，并更新 todo 相关计数。

    这个节点负责两层工作：
    - 执行模型请求的 LangChain tools，并把结果包装成 ToolMessage
    - 维护 s03 特有的 todo reminder 逻辑

    具体行为：
    - 在终端打印工具名和前 200 个字符结果，方便观察 agent 行动
    - 如果本轮调用了 todo，把 rounds_since_todo 清零
    - 如果本轮没有调用 todo，把 rounds_since_todo 加一
    - 连续 3 轮没调用 todo 时，插入提醒 HumanMessage
    """
    last = state["messages"][-1]

    if not isinstance(last, AIMessage):
        return {}

    tool_calls = last.tool_calls or []
    used_todo = any(call["name"] == "todo" for call in tool_calls)

    new_messages = []

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call.get("args", {})
        tool_obj = TOOL_BY_NAME.get(tool_name)

        if tool_obj is None:
            output = f"Unknown tool: {tool_name}"
        else:
            try:
                # @tool 包装后的函数要用 invoke 调用。
                # 这样会走 LangChain Tool 的参数校验和执行路径。
                output = tool_obj.invoke(tool_args)
            except Exception as e:
                output = f"Error: {e}"

        # 打印一个简短 trace，效果接近原版 s03 的：
        #   > tool_name:
        #   tool output...
        print(f"> {call['name']}:")
        print(str(output)[:200])

        # ToolMessage 必须带上 tool_call_id。
        # 这个 id 用来告诉模型：这条工具结果对应刚才哪个 tool_call。
        new_messages.append(
            ToolMessage(
                content=str(output),
                tool_call_id=call["id"],
            )
        )

    rounds_since_todo = 0 if used_todo else state.get("rounds_since_todo", 0) + 1

    # 原版 s03 把 reminder 和 tool_result 放在同一个 user turn 里。
    #
    # 在 LangChain 的 tool calling 协议里，AIMessage(tool_calls=[...]) 后面必须
    # 紧跟对应的 ToolMessage。这里先追加所有 ToolMessage，再追加一个 HumanMessage，
    # 既满足协议，也能让模型在下一轮调用前看到提醒。
    if rounds_since_todo >= 3:
        new_messages.append(HumanMessage(content="<reminder>Update your todos.</reminder>"))

    return {
        "messages": new_messages,
        "rounds_since_todo": rounds_since_todo,
    }


# ------------------------------------------------------------
# 8. 构建 LangGraph
# ------------------------------------------------------------

builder = StateGraph(AgentState)

# START -> agent
#
# 每次 graph.invoke(...) 都先进入 agent 节点，让模型决定下一步。
builder.add_edge(START, "agent")

# agent 节点：调用模型。
builder.add_node("agent", agent_node)

# tools 节点：执行工具，并维护 todo reminder 状态。
builder.add_node("tools", tools_node)

# agent -> tools 或 agent -> END
#
# tools_condition 会检查 agent 刚刚返回的 AIMessage：
# - 如果有 tool_calls，返回 "tools"
# - 如果没有 tool_calls，返回 "__end__"
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "__end__",
    },
)

# tools -> agent
#
# 工具执行结果追加到 messages 后，需要回到模型，让模型基于结果继续行动或结束。
builder.add_edge("tools", "agent")


# ------------------------------------------------------------
# 9. 编译 Graph
# ------------------------------------------------------------

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


# ------------------------------------------------------------
# 10. 对外调用封装
# ------------------------------------------------------------

def run_once(query: str, thread_id: str = "default") -> str:
    """
    执行一次用户输入，并返回最终模型文本。

    由于 graph 使用了 MemorySaver，同一个 thread_id 会保留历史消息，
    因此连续调用 run_once(..., thread_id="cli-session") 就是多轮对话。
    """
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "rounds_since_todo": 0,
        },
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
        },
    )

    last = result["messages"][-1]

    return last.content if isinstance(last.content, str) else str(last.content)


def agent_loop(messages: list[BaseMessage]) -> None:
    """
    兼容原版 s03 的 agent_loop(messages) 调用方式。

    这种形式适合外部自己维护 messages 列表。
    为了避免 MemorySaver 把同一个 messages 列表重复追加进 checkpoint，
    这里根据 messages 对象和当前长度生成一个临时 thread_id。
    """
    final_state = graph.invoke(
        {
            "messages": messages,
            "rounds_since_todo": 0,
        },
        config={
            "configurable": {"thread_id": f"agent-loop-{id(messages)}-{len(messages)}"},
            "recursion_limit": 100,
        },
    )

    messages[:] = final_state["messages"]


# ------------------------------------------------------------
# 11. CLI 入口
# ------------------------------------------------------------

if __name__ == "__main__":
    thread_id = "cli-session"

    while True:
        try:
            query = input("\033[36mgraph-s03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        answer = run_once(query, thread_id=thread_id)
        print(answer)
        print()
