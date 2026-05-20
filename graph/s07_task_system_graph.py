#!/usr/bin/env python3
"""
s07_task_system_graph.py - 使用 LangGraph 改写持久任务系统


s07 的关键点：任务不是只放在对话上下文里，而是持久化为 .tasks/task_*.json。
这样即使上下文被压缩，任务状态和依赖图仍然保存在文件系统中。
"""

import json
import os
import subprocess
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
TASKS_DIR = WORKDIR / ".tasks"
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")
SYSTEM = f"你是位于 {WORKDIR} 的编程 agent。请使用任务工具规划、跟踪和更新工作。"


class TaskManager:
    """基于 JSON 文件的任务管理器，支持状态和 blockedBy 依赖。"""

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, task: dict) -> None:
        self._path(task["id"]).write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str | None = None,
        addBlockedBy: list[int] | None = None,
        removeBlockedBy: list[int] | None = None,
    ) -> str:
        task = self._load(task_id)

        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)

        if addBlockedBy:
            task["blockedBy"] = sorted(set(task["blockedBy"] + addBlockedBy))

        if removeBlockedBy:
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in removeBlockedBy]

        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int) -> None:
        """任务完成后，从其他任务的 blockedBy 中移除它。"""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text(encoding="utf-8"))
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        tasks = []
        files = sorted(self.dir.glob("task_*.json"), key=lambda f: int(f.stem.split("_")[1]))
        for f in files:
            tasks.append(json.loads(f.read_text(encoding="utf-8")))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)


def safe_path(p: str) -> Path:
    """限制文件工具只能访问当前工作区。"""
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


@tool("task_create")
def task_create(subject: str, description: str = "") -> str:
    """创建持久任务。"""
    return TASKS.create(subject, description)


@tool("task_update")
def task_update(
    task_id: int,
    status: str | None = None,
    addBlockedBy: list[int] | None = None,
    removeBlockedBy: list[int] | None = None,
) -> str:
    """更新任务状态或依赖。"""
    return TASKS.update(task_id, status, addBlockedBy, removeBlockedBy)


@tool("task_list")
def task_list() -> str:
    """列出所有任务。"""
    return TASKS.list_all()


@tool("task_get")
def task_get(task_id: int) -> str:
    """查看单个任务详情。"""
    return TASKS.get(task_id)


TOOLS = [run_bash, run_read, run_write, run_edit, task_create, task_update, task_list, task_get]
TOOL_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    """
    LangGraph 状态。

    s07 不需要整体替换历史，因此使用 add_messages：
    agent 节点返回 AIMessage，tools 节点返回 ToolMessage，LangGraph 会自动追加。
    任务本身不保存在 messages 里，而是持久化到 .tasks/，这是 s07 的重点。
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


def agent_node(state: AgentState) -> dict:
    """调用模型。"""
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行工具调用并生成 ToolMessage。

    task_create / task_update / task_list / task_get 都会访问 .tasks/。
    因此即使对话被压缩或程序重启，任务文件仍能保留下来。
    """
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
builder.add_edge(START, "agent")
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "agent")
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
            query = input("\033[36mgraph-s07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        print(run_once(query, thread_id))
        print()
