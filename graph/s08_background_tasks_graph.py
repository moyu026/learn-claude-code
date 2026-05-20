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

WORKDIR = Path.cwd()
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")
SYSTEM = f"你是位于 {WORKDIR} 的编程 agent。遇到长时间运行的命令时，请使用 background_run。"


class BackgroundManager:
    """后台线程管理器：启动命令、保存状态、排队完成通知。"""

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self._notification_queue: list[dict] = []
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """启动后台线程，立即返回 task_id。"""
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(target=self._execute, args=(task_id, command), daemon=True)
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        """后台线程函数：执行命令，写入状态，并把完成通知放进队列。"""
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=300)
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        final_output = output or "(no output)"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = final_output

        with self._lock:
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
            task = self.tasks.get(task_id)
            if not task:
                return f"Error: Unknown task {task_id}"
            return f"[{task['status']}] {task['command'][:60]}\n{task.get('result') or '(running)'}"

        lines = [f"{tid}: [{t['status']}] {t['command'][:60]}" for tid, t in self.tasks.items()]
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list[dict]:
        """取出并清空所有未发送给模型的完成通知。"""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


BG = BackgroundManager()


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool("bash")
def run_bash(command: str) -> str:
    """执行阻塞式 shell 命令。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """读取工作区文件。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """写入工作区文件。"""
    try:
        fp = safe_path(path)
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
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("background_run")
def background_run(command: str) -> str:
    """在后台线程运行命令，立即返回任务 id。"""
    return BG.run(command)


@tool("check_background")
def check_background(task_id: str | None = None) -> str:
    """检查后台任务状态；不传 task_id 时列出所有任务。"""
    return BG.check(task_id)


TOOLS = [run_bash, run_read, run_write, run_edit, background_run, check_background]
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
llm_with_tools = llm.bind_tools(TOOLS)


def inject_notifications_node(state: AgentState) -> dict:
    """
    模型调用前排空后台通知队列，并作为 HumanMessage 注入上下文。

    这对应原版 s08 的 “drain queue before each LLM call”：
    后台线程完成后不会直接打断模型，而是在下一轮模型调用前统一交付结果。
    """
    notifs = BG.drain_notifications()
    if not notifs:
        return {}
    text = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
    return {"messages": [HumanMessage(content=f"<background-results>\n{text}\n</background-results>")]}


def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    results = []
    for call in last.tool_calls or []:
        name = call["name"]
        tool_obj = TOOL_BY_NAME.get(name)
        try:
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=name))
    return {"messages": results}


builder = StateGraph(AgentState)
builder.add_edge(START, "notify")
builder.add_node("notify", inject_notifications_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge("notify", "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "notify")
graph = builder.compile(checkpointer=MemorySaver())


def message_text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def run_once(query: str, thread_id: str = "default") -> str:
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100},
    )
    return message_text(final_state["messages"][-1])


def agent_loop(messages: list[BaseMessage]) -> None:
    final_state = graph.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": f"agent-loop-{id(messages)}-{len(messages)}"}, "recursion_limit": 100},
    )
    messages[:] = final_state["messages"]


if __name__ == "__main__":
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
