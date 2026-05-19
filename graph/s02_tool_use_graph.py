import os
import subprocess
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

# LangChain 的消息类型
# BaseMessage 是所有消息的基类
# HumanMessage 表示用户消息
# SystemMessage 表示系统提示词
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# @tool 用来把普通 Python 函数包装成 LangChain Tool
# 包装后，函数名、参数、docstring 会被转成工具 schema
from langchain_core.tools import tool

# OpenAI-compatible 模型入口
# 只要你的服务兼容 OpenAI Chat Completions / Responses 风格，一般可以通过 base_url 接入
from langchain_openai import ChatOpenAI

# LangGraph 的图结构
# StateGraph：定义有状态工作流
# START：图的起点
from langgraph.graph import StateGraph, START

# add_messages 是 LangGraph 提供的 reducer
# 它的作用是：当节点返回 {"messages": [new_message]} 时，
# 不会覆盖原 messages，而是追加到原 messages 后面
from langgraph.graph.message import add_messages

# ToolNode：LangGraph 预置工具节点
# 它会自动读取 AIMessage 中的 tool_calls，并执行对应工具
#
# tools_condition：LangGraph 预置条件函数
# 它会检查最后一条 AIMessage：
# - 如果有 tool_calls，路由到 tools 节点
# - 如果没有 tool_calls，路由到 __end__
from langgraph.prebuilt import ToolNode, tools_condition

# MemorySaver 是内存版 checkpoint
# 用于让同一个 thread_id 的多轮调用共享历史状态
from langgraph.checkpoint.memory import MemorySaver

# ------------------------------------------------------------
# 1. 加载环境变量
# ------------------------------------------------------------

load_dotenv(override=True)

# 当前工作目录，等价于你原来的 WORKDIR = Path.cwd()
# 所有文件读写和 shell 命令都限制在这个目录下执行
WORKDIR = Path.cwd()

# 模型名从环境变量读取
# 例如：
# MODEL_ID=gpt-4.1
# MODEL_ID=qwen3-coder-480b-a35b-instruct-fc
MODEL = os.environ["MODEL_ID"]

# 系统提示词
# 这里和你原来的 Anthropic 版本保持一致
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use tools to solve tasks. Act, don't explain."
)


# ------------------------------------------------------------
# 2. 路径安全检查
# ------------------------------------------------------------

def safe_path(p: str) -> Path:
    """
    把用户/模型传入的相对路径转换成绝对路径，并限制它不能逃逸出 WORKDIR。

    例如：
    - "a.txt"                  -> 允许
    - "src/main.py"            -> 允许
    - "../secret.txt"          -> 禁止
    - "/etc/passwd"            -> 禁止

    这个函数很重要，因为 agent 可以调用 write_file/edit_file。
    如果没有路径检查，模型可能误操作工作区外的文件。
    """
    path = (WORKDIR / p).resolve()

    # Python 3.9+ 支持 Path.is_relative_to
    # 表示 path 是否在 WORKDIR 内部
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")

    return path


# ------------------------------------------------------------
# 3. 定义工具
# ------------------------------------------------------------
#
# 原来的 Anthropic 写法是：
#
# TOOL_HANDLERS = {
#     "bash": lambda **kw: run_bash(kw["command"]),
# }
#
# TOOLS = [
#     {"name": "bash", "description": "...", "input_schema": {...}}
# ]
#
# 在 LangChain / LangGraph 里，更常见的做法是：
#
# @tool
# def run_bash(...):
#     """工具描述"""
#
# 然后把这些 tool 函数交给：
# - llm.bind_tools(TOOLS)
# - ToolNode(TOOLS)
#
# bind_tools 负责告诉模型有哪些工具；
# ToolNode 负责真正执行工具。


@tool("bash")
def run_bash(command: str) -> str:
    """
    Run a shell command in the current workspace.

    这个 docstring 会进入工具描述。
    模型会根据这个描述判断什么时候调用 bash 工具。
    """
    # 非常简单的危险命令拦截
    # 注意：这不是完整沙箱，只是基础防护
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # shell=True 方便模型生成复合命令
        # cwd=WORKDIR 限制命令在当前项目目录下执行
        # capture_output=True 捕获 stdout/stderr
        # timeout=120 防止命令卡死
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # 合并标准输出和错误输出
        out = (r.stdout + r.stderr).strip()

        # 限制返回长度，避免工具输出把上下文撑爆
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except Exception as e:
        return f"Error: {e}"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """
    Read file contents from the current workspace.

    参数：
    - path: 文件路径，相对于 WORKDIR
    - limit: 可选，只读取前 N 行
    """
    try:
        fp = safe_path(path)

        # 加 encoding="utf-8" 更稳定
        # Windows 下如果遇到非 utf-8 文件，可能需要改成 errors="ignore"
        text = fp.read_text(encoding="utf-8")

        lines = text.splitlines()

        # 如果传了 limit，并且文件行数超过 limit，就只返回前 limit 行
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """
    Write content to a file in the current workspace.

    注意：
    - 如果父目录不存在，会自动创建
    - 如果文件已存在，会整体覆盖
    """
    try:
        fp = safe_path(path)

        # 确保父目录存在
        fp.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        fp.write_text(content, encoding="utf-8")

        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file in the current workspace.

    这个工具只替换第一次出现的 old_text。
    适合让 agent 做精确局部编辑。
    """
    try:
        fp = safe_path(path)

        content = fp.read_text(encoding="utf-8")

        # 这里要求 old_text 必须完全匹配
        # 好处是可控；坏处是模型给出的 old_text 稍有差异就会失败
        if old_text not in content:
            return f"Error: Text not found in {path}"

        # 只替换一次，避免误改多个相似片段
        new_content = content.replace(old_text, new_text, 1)

        fp.write_text(new_content, encoding="utf-8")

        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


# 所有工具统一放到一个列表里
# 这个列表会同时给：
# 1. llm.bind_tools(TOOLS)
# 2. ToolNode(TOOLS)
TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
]


# ------------------------------------------------------------
# 4. 定义 LangGraph State
# ------------------------------------------------------------

class AgentState(TypedDict):
    """
    LangGraph 的共享状态。

    这里最核心的是 messages。

    messages 里保存：
    - HumanMessage：用户输入
    - AIMessage：模型输出
    - ToolMessage：工具执行结果

    Annotated[list[BaseMessage], add_messages] 的含义是：

    当某个节点返回：
        {"messages": [new_message]}

    LangGraph 不会把原来的 messages 覆盖掉，
    而是调用 add_messages，把 new_message 追加进去。

    这就等价于你原来手写的：

        messages.append(...)
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------
# 5. 初始化模型
# ------------------------------------------------------------

llm = ChatOpenAI(
    # 模型名
    model=MODEL,

    # OpenAI API Key
    # 如果你是公司内部 OpenAI-compatible 服务，有些服务不校验 key，
    # 可以用 EMPTY 占位
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),

    # OpenAI-compatible base url
    # 推荐统一用 OPENAI_BASE_URL
    # 兼容你之前可能使用的 LLM_BASE_URL
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),

    # coding agent 通常建议 temperature=0，减少随机性
    temperature=0,

    # 最大输出 token
    max_tokens=8000,
)

# 把工具绑定给模型
#
# 注意：
# bind_tools 不是执行工具。
# 它只是告诉模型：
# “你可以调用这些工具，工具名、参数 schema、描述如下。”
#
# 真正执行工具的是后面的 ToolNode。
llm_with_tools = llm.bind_tools(TOOLS)


# ------------------------------------------------------------
# 6. 定义 agent 节点
# ------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    agent 节点负责调用 LLM。

    输入：
        state，其中包含历史 messages

    输出：
        {"messages": [response]}

    LangGraph 会通过 add_messages 把 response 追加到历史消息中。
    """

    # 每次调用模型时，都把 SystemMessage 放在最前面
    #
    # 为什么不把 SystemMessage 存进 state["messages"]？
    # 也可以存，但实践中常见做法是：
    # - system prompt 固定由节点注入
    # - state 只保存用户、模型、工具之间的交互历史
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM)] + state["messages"]
    )

    # response 通常是 AIMessage
    #
    # 如果模型决定调用工具，response.tool_calls 会有内容
    # 如果模型不调用工具，response.content 就是最终回答
    return {"messages": [response]}


# ------------------------------------------------------------
# 7. 定义工具节点
# ------------------------------------------------------------

# ToolNode 会自动做这些事：
#
# 1. 查看 state["messages"][-1]
# 2. 读取最后一条 AIMessage 里的 tool_calls
# 3. 根据 tool_calls 中的 name 找到对应工具
# 4. 用 tool_calls 中的 args 执行工具
# 5. 把工具结果包装成 ToolMessage
# 6. 返回 {"messages": [ToolMessage(...), ...]}
#
# 所以你不再需要自己写：
#
# for block in response.content:
#     if block.type == "tool_use":
#         handler = TOOL_HANDLERS.get(block.name)
#         output = handler(**block.input)
tool_node = ToolNode(TOOLS)


# ------------------------------------------------------------
# 8. 构建 LangGraph
# ------------------------------------------------------------

# 创建一个有状态图
builder = StateGraph(AgentState)

# 添加 agent 节点
builder.add_node("agent", agent_node)

# 添加 tools 节点
builder.add_node("tools", tool_node)

# START -> agent
#
# 每次 graph.invoke(...) 都会从 START 开始，
# 先进入 agent 节点，让模型判断要不要调用工具。
builder.add_edge(START, "agent")

# agent -> tools 或 agent -> END
#
# tools_condition 会检查 agent 节点刚刚返回的 AIMessage：
#
# - 如果 AIMessage 里有 tool_calls：
#       返回 "tools"
#
# - 如果没有 tool_calls：
#       返回 "__end__"
#
# 这就替代了你原来的：
#
# if response.stop_reason != "tool_use":
#     return
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
# 工具执行完以后，工具结果会变成 ToolMessage 追加到 messages。
# 然后再回到 agent，让模型基于工具结果继续思考。
#
# 这就替代了你原来的：
#
# messages.append({"role": "user", "content": results})
# 然后 while True 继续下一轮 client.messages.create(...)
builder.add_edge("tools", "agent")


# ------------------------------------------------------------
# 9. 编译图
# ------------------------------------------------------------

# MemorySaver 是内存级 checkpoint
#
# 它允许同一个 thread_id 在多次 graph.invoke(...) 之间保留 messages。
#
# 如果没有 checkpointer：
# 每次 graph.invoke({"messages": [HumanMessage(...)]}) 都是一个新会话。
#
# 有了 MemorySaver + thread_id：
# 多次调用 run_once(..., thread_id="cli-session") 会共享历史上下文。
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


# ------------------------------------------------------------
# 10. 单轮调用封装
# ------------------------------------------------------------

def run_once(query: str, thread_id: str = "default") -> str:
    """
    执行一次用户输入。

    由于 graph 使用了 checkpointer，
    只要 thread_id 相同，LangGraph 就会自动接上之前的消息历史。

    参数：
    - query: 用户输入
    - thread_id: 会话 ID

    返回：
    - 最后一条 AIMessage 的文本内容
    """

    result = graph.invoke(
        # 新增一条用户消息
        {"messages": [HumanMessage(content=query)]},

        # thread_id 是 LangGraph checkpoint 的关键
        # 同一个 thread_id 表示同一个会话
        config={"configurable": {"thread_id": thread_id}},
    )

    # result 是最终 state
    # result["messages"] 是完整消息历史
    last = result["messages"][-1]

    # 一般情况下，最后一条消息是 AIMessage，content 是字符串
    # 某些模型/适配器可能返回结构化 content，所以这里做一下兼容
    return last.content if isinstance(last.content, str) else str(last.content)


# ------------------------------------------------------------
# 11. CLI 入口
# ------------------------------------------------------------

if __name__ == "__main__":
    # 固定 thread_id，保证这个 CLI 进程内的多轮输入共享上下文
    thread_id = "cli-session"

    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 输入 q / exit / 空字符串则退出
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 调用 LangGraph
        answer = run_once(query, thread_id=thread_id)

        # 打印最终回答
        print(answer)
        print()
