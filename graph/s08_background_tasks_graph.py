#!/usr/bin/env python3
"""
s08_background_tasks_graph.py - 使用 LangGraph 改写后台任务示例

s08 的关键点：长命令用后台线程运行，agent 不阻塞等待。
后台任务完成后，结果进入通知队列；下一次模型调用前，图会把通知注入 messages。
"""

import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)

# 当前示例默认在启动脚本的目录中执行命令和读写文件。
# 这样 agent 看到的“工作区”与用户运行脚本的位置一致。
WORKDIR = Path.cwd()

# 模型名优先读取 OPENAI_MODEL，其次兼容仓库里其他示例使用的 MODEL_ID。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# 系统提示只给 lead agent 使用。这里明确告诉模型：
# 普通短命令可以用 bash，耗时命令应该交给 background_run，避免阻塞整轮图执行。
SYSTEM = f"你是位于 {WORKDIR} 的编程 agent。遇到长时间运行的命令时，请使用 background_run。"


class BackgroundManager:
    """后台线程管理器：启动命令、保存状态、排队完成通知。"""

    def __init__(self):
        # tasks 保存所有后台任务的最终权威状态。
        # key 是短 task_id，value 中记录命令、运行状态和完整结果。
        self.tasks: dict[str, dict] = {}

        # 通知队列只保存“已经完成、尚未注入给模型”的任务摘要。
        # 它和 tasks 分开，是为了让模型只在下一轮看到新完成的任务，
        # 而不是每次都把所有历史后台任务塞进上下文。
        self._notification_queue: list[dict] = []

        # 后台线程会写通知队列，LangGraph 主线程会 drain 通知队列；
        # 因此队列读写需要加锁，避免并发修改导致遗漏或重复。
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """启动后台线程，立即返回 task_id。"""
        # 取 UUID 前 8 位作为用户可读的任务 id，足够演示且便于复制查询。
        task_id = str(uuid.uuid4())[:8]

        # 先登记为 running，再启动线程。
        # 这样即使线程刚启动就执行很久，用户也能通过 check_background 查到它。
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}

        # daemon=True 表示主进程退出时不强行等待后台线程；
        # 适合这个交互式教学示例，避免用户 Ctrl-C 后进程无法退出。
        thread = threading.Thread(target=self._execute, args=(task_id, command), daemon=True)
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        """后台线程函数：执行命令，写入状态，并把完成通知放进队列。"""
        try:
            # 后台命令允许比普通 bash 更长的运行时间。
            # capture_output=True 把 stdout/stderr 都收集起来，稍后统一交回模型。
            r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=300)

            # 输出最多保留 50000 字符，避免超长日志撑爆模型上下文。
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        final_output = output or "(no output)"

        # 任务表保存完整结果，供用户或 agent 之后用 check_background 查询。
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = final_output

        with self._lock:
            # 通知队列只放短摘要：它会被自动注入到下一次模型调用。
            # result 截断到 500 字符，是为了提醒模型“任务完成了”以及大致结果；
            # 如果模型需要完整日志，可以再主动调用 check_background(task_id)。
            self._notification_queue.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "command": command[:80],
                    "result": final_output[:500],
                }
            )

    def check(self, task_id: str | None = None) -> str:
        """查看单个后台任务，或列出所有后台任务。"""
        if task_id:
            # 查询单个任务时返回状态、命令片段和完整结果。
            # 任务还在运行时 result 为 None，这里显示为 (running)。
            task = self.tasks.get(task_id)
            if not task:
                return f"Error: Unknown task {task_id}"
            return f"[{task['status']}] {task['command'][:60]}\n{task.get('result') or '(running)'}"

        # 不传 task_id 时只列摘要，方便模型或用户快速知道当前有哪些后台任务。
        lines = [f"{tid}: [{t['status']}] {t['command'][:60]}" for tid, t in self.tasks.items()]
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list[dict]:
        """取出并清空所有未发送给模型的完成通知。"""
        with self._lock:
            # drain 语义：一次性取走当前所有通知，并清空队列。
            # 这样同一条完成通知只会注入模型上下文一次。
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


# 全局后台任务管理器。
# 这个示例是单进程运行，所以用内存对象即可；重启脚本后后台任务状态会丢失。
BG = BackgroundManager()


def safe_path(p: str) -> Path:
    """把用户给的相对路径限制在当前工作区内。"""
    # resolve 会规整 ..、符号链接等路径片段；
    # 后面的 is_relative_to 用来阻止读取或写入工作区之外的文件。
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool("bash")
def run_bash(command: str) -> str:
    """执行阻塞式 shell 命令。"""
    # 这是同步工具：LangGraph 会等待命令结束后才继续下一步。
    # 因此它适合短命令；耗时命令应该让模型选择 background_run。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # timeout=120 是同步命令的硬上限，防止 agent 卡死在一次工具调用里。
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """读取工作区文件。"""
    try:
        # safe_path 保证模型不能通过 ../ 读取工作区外的敏感文件。
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            # limit 用于只读取文件开头，减少大文件对上下文窗口的占用。
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """写入工作区文件。"""
    try:
        fp = safe_path(path)
        # 自动创建父目录，让模型可以直接写入新路径。
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """替换文件中的第一处精确文本。"""
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            # 精确匹配失败时不做模糊编辑，避免模型误改相似代码。
            return f"Error: Text not found in {path}"
        # 只替换第一处，降低一次工具调用造成大范围改动的风险。
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("background_run")
def background_run(command: str) -> str:
    """在后台线程运行命令，立即返回任务 id。"""
    # 这个工具不会等待命令完成。
    # 命令结束后，BackgroundManager 会把完成摘要放入通知队列。
    return BG.run(command)


@tool("check_background")
def check_background(task_id: str | None = None) -> str:
    """检查后台任务状态；不传 task_id 时列出所有任务。"""
    # 模型收到后台完成摘要后，如果摘要被截断，可以用这个工具取完整结果。
    return BG.check(task_id)


# TOOLS 是暴露给模型的工具列表；bind_tools 会根据每个 @tool 函数生成工具 schema。
TOOLS = [run_bash, run_read, run_write, run_edit, background_run, check_background]

# 执行工具调用时需要根据模型返回的工具名找到对应工具对象。
TOOL_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    """
    LangGraph 状态。

    后台任务的真实状态不放在 messages 中，而是放在 BG.tasks 和通知队列里。
    messages 只接收完成通知的快照，避免每轮都把所有后台任务状态塞进上下文。
    """
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)

# 绑定工具后的模型会在需要时返回 tool_calls，而不是直接给最终回答。
llm_with_tools = llm.bind_tools(TOOLS)


def inject_notifications_node(state: AgentState) -> dict:
    """
    模型调用前排空后台通知队列，并作为 HumanMessage 注入上下文。

    这对应原版 s08 的 “drain queue before each LLM call”：
    后台线程完成后不会直接打断模型，而是在下一轮模型调用前统一交付结果。
    """
    notifs = BG.drain_notifications()
    if not notifs:
        # 返回空 dict 表示该节点不修改图状态。
        return {}

    # 把后台完成通知包装在 XML 风格标签中，降低它和用户普通消息混淆的概率。
    text = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
    return {"messages": [HumanMessage(content=f"<background-results>\n{text}\n</background-results>")]}


def agent_node(state: AgentState) -> dict:
    """调用带工具能力的模型，让模型决定回答或继续请求工具。"""
    # 每次调用都重新注入 SystemMessage，而历史对话来自 LangGraph state。
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])

    # LangGraph 节点返回 dict；messages 会通过 add_messages 追加到已有消息列表。
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """执行模型刚刚请求的工具调用，并把结果转成 ToolMessage。"""
    # LangGraph 约定：如果 agent 节点返回了 tool_calls，这些调用会挂在最后一条 AIMessage 上。
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    results = []
    for call in last.tool_calls or []:
        name = call["name"]
        tool_obj = TOOL_BY_NAME.get(name)
        try:
            # LangChain tool 的 invoke 接收参数 dict，会负责按函数签名映射参数。
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            # 工具异常也作为工具结果返回给模型，让模型可以自行恢复或解释错误。
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])

        # ToolMessage 必须带上 tool_call_id，这样模型能把结果对应回刚才的具体工具调用。
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=name))
    return {"messages": results}


builder = StateGraph(AgentState)

# 图的主循环：
# START -> notify -> agent -> tools -> notify -> ...
#
# notify：把已经完成的后台任务结果注入 messages。
# agent：调用模型，模型可能直接回答，也可能提出工具调用。
# tools：执行工具并把结果作为 ToolMessage 追加回 messages。
builder.add_edge(START, "notify")
builder.add_node("notify", inject_notifications_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge("notify", "agent")

# tools_condition 是 LangGraph 预置条件：
# - 如果最后一条 AIMessage 有 tool_calls，则进入 tools 节点；
# - 如果没有工具调用，则结束本轮图执行。
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})

# 工具执行完回到 notify，而不是直接回 agent。
# 这样如果后台任务恰好在工具执行期间完成，下一次模型调用前也能收到通知。
builder.add_edge("tools", "notify")

# MemorySaver 用内存保存 thread_id 对应的对话状态。
# 同一个 thread_id 多次 run_once 会延续上下文；换 thread_id 则是新会话。
graph = builder.compile(checkpointer=MemorySaver())


def message_text(message: BaseMessage) -> str:
    """把 LangChain message content 规整成字符串，便于 CLI 打印。"""
    return message.content if isinstance(message.content, str) else str(message.content)


def run_once(query: str, thread_id: str = "default") -> str:
    """执行一次用户输入，并返回本轮最终消息文本。"""
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100},
    )
    return message_text(final_state["messages"][-1])


def agent_loop(messages: list[BaseMessage]) -> None:
    """兼容其他示例的 agent_loop 接口：传入消息列表，并原地更新为图执行后的状态。"""
    final_state = graph.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": f"agent-loop-{id(messages)}-{len(messages)}"}, "recursion_limit": 100},
    )
    messages[:] = final_state["messages"]


if __name__ == "__main__":
    # CLI 模式固定使用同一个 thread_id，因此多轮输入会共享 LangGraph 记忆。
    thread_id = "cli-session"
    while True:
        try:
            query = input("\033[36mgraph-s08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        print(run_once(query, thread_id))
        print()
