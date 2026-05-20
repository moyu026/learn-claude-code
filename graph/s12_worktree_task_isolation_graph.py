#!/usr/bin/env python3
"""
s12_worktree_task_isolation_graph.py - 使用 LangGraph 改写 Worktree + Task Isolation

s12 的关键点：
- .tasks 是任务控制面
- .worktrees 是执行隔离面
- 每个任务可以绑定一个 git worktree
- worktree 生命周期事件写入 .worktrees/events.jsonl
"""

import json
import os
import re
import subprocess
import time
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


def detect_repo_root(cwd: Path) -> Path | None:
    """如果当前目录在 git 仓库内，返回仓库根目录。"""
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR
SYSTEM = (
    f"你是位于 {WORKDIR} 的编程 agent。"
    "请使用 task + worktree 工具处理多任务工作。"
    "遇到并行任务或有风险的修改时，先创建任务，再分配 worktree 执行通道，"
    "在对应通道中运行命令，最后根据结果选择 keep 或 remove。"
    "需要查看生命周期时，请调用 worktree_events。"
)


class EventBus:
    """append-only lifecycle event log。"""

    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, task: dict | None = None, worktree: dict | None = None, error: str | None = None) -> None:
        payload = {"event": event, "ts": time.time(), "task": task or {}, "worktree": worktree or {}}
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        items = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2, ensure_ascii=False)


class TaskManager:
    """持久任务板，并支持 worktree 绑定。"""

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
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
        task["updated_at"] = time.time()
        self._path(task["id"]).write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str | None = None, owner: str | None = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            task["owner"] = owner
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "in_progress"
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task["worktree"] = ""
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def list_all(self) -> str:
        tasks = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(self.dir.glob("task_*.json"))]
        if not tasks:
            return "No tasks."
        lines = []
        for task in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task["status"], "[?]")
            owner = f" owner={task['owner']}" if task.get("owner") else ""
            wt = f" wt={task['worktree']}" if task.get("worktree") else ""
            lines.append(f"{marker} #{task['id']}: {task['subject']}{owner}{wt}")
        return "\n".join(lines)


TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


class WorktreeManager:
    """管理 git worktree 和 .worktrees/index.json。"""

    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2), encoding="utf-8")
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        try:
            r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=self.repo_root, capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(["git", *args], cwd=self.repo_root, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError((r.stdout + r.stderr).strip() or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, data: dict) -> None:
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find(self, name: str) -> dict | None:
        for wt in self._load_index().get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError("Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -")

    def create(self, name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
        # 创建 worktree 是 s12 的核心动作。
        #
        # 它同时触碰三类状态：
        # 1. git worktree：真正创建隔离目录和分支
        # 2. .worktrees/index.json：记录这个执行通道
        # 3. .tasks/task_*.json：可选，把任务绑定到该 worktree
        #
        # 每个阶段都会写 lifecycle event，便于事后审计。
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name
        branch = f"wt/{name}"
        self.events.emit("worktree.create.before", task={"id": task_id} if task_id is not None else {}, worktree={"name": name, "base_ref": base_ref})
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])
            entry = {"name": name, "path": str(path), "branch": branch, "task_id": task_id, "status": "active", "created_at": time.time()}
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)
            self.events.emit("worktree.create.after", task={"id": task_id} if task_id is not None else {}, worktree=entry)
            return json.dumps(entry, indent=2, ensure_ascii=False)
        except Exception as e:
            self.events.emit("worktree.create.failed", task={"id": task_id} if task_id is not None else {}, worktree={"name": name}, error=str(e))
            raise

    def list_all(self) -> str:
        items = self._load_index().get("worktrees", [])
        if not items:
            return "No worktrees in index."
        return "\n".join(f"[{wt.get('status', 'unknown')}] {wt['name']} -> {wt['path']} ({wt.get('branch', '-')})" for wt in items)

    def status(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(["git", "status", "--short", "--branch"], cwd=path, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip() or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        try:
            r = subprocess.run(command, shell=True, cwd=path, capture_output=True, text=True, timeout=300)
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def keep(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)
        self.events.emit("worktree.keep", task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {}, worktree={"name": name, "status": "kept"})
        return json.dumps(kept, indent=2, ensure_ascii=False) if kept else f"Error: Unknown worktree '{name}'"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        # 删除 worktree 时不直接从 index 移除记录，而是标记为 removed。
        # 这样可以保留生命周期历史，知道某个任务曾经使用过哪个执行通道。
        #
        # complete_task=True 时，会顺带把绑定任务标记为 completed 并解绑 worktree。
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        self.events.emit("worktree.remove.before", task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {}, worktree={"name": name})
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)
            if complete_task and wt.get("task_id") is not None:
                self.tasks.update(wt["task_id"], status="completed")
                self.tasks.unbind_worktree(wt["task_id"])
                self.events.emit("task.completed", task={"id": wt["task_id"], "status": "completed"}, worktree={"name": name})
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)
            self.events.emit("worktree.remove.after", task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {}, worktree={"name": name, "status": "removed"})
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit("worktree.remove.failed", task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {}, worktree={"name": name}, error=str(e))
            raise


WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool("bash")
def bash(command: str) -> str:
    """在当前工作区执行阻塞命令。"""
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
def read_file(path: str, limit: int | None = None) -> str:
    """读取当前工作区文件。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def write_file(path: str, content: str) -> str:
    """写入当前工作区文件。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def edit_file(path: str, old_text: str, new_text: str) -> str:
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
    """创建任务。"""
    return TASKS.create(subject, description)


@tool("task_list")
def task_list() -> str:
    """列出任务。"""
    return TASKS.list_all()


@tool("task_get")
def task_get(task_id: int) -> str:
    """查看任务。"""
    return TASKS.get(task_id)


@tool("task_update")
def task_update(task_id: int, status: str | None = None, owner: str | None = None) -> str:
    """更新任务状态或 owner。"""
    return TASKS.update(task_id, status, owner)


@tool("task_bind_worktree")
def task_bind_worktree(task_id: int, worktree: str, owner: str = "") -> str:
    """把任务绑定到 worktree。"""
    return TASKS.bind_worktree(task_id, worktree, owner)


@tool("worktree_create")
def worktree_create(name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
    """创建 git worktree。"""
    return WORKTREES.create(name, task_id, base_ref)


@tool("worktree_list")
def worktree_list() -> str:
    """列出 worktree index。"""
    return WORKTREES.list_all()


@tool("worktree_status")
def worktree_status(name: str) -> str:
    """查看 worktree git 状态。"""
    return WORKTREES.status(name)


@tool("worktree_run")
def worktree_run(name: str, command: str) -> str:
    """在指定 worktree 中运行命令。"""
    return WORKTREES.run(name, command)


@tool("worktree_keep")
def worktree_keep(name: str) -> str:
    """标记 worktree 为 kept。"""
    return WORKTREES.keep(name)


@tool("worktree_remove")
def worktree_remove(name: str, force: bool = False, complete_task: bool = False) -> str:
    """删除 worktree，可选完成绑定任务。"""
    return WORKTREES.remove(name, force, complete_task)


@tool("worktree_events")
def worktree_events(limit: int = 20) -> str:
    """查看最近 lifecycle events。"""
    return EVENTS.list_recent(limit)


TOOLS = [
    bash, read_file, write_file, edit_file,
    task_create, task_list, task_get, task_update, task_bind_worktree,
    worktree_create, worktree_list, worktree_status, worktree_run, worktree_keep, worktree_remove, worktree_events,
]
TOOL_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
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
    """
    调用模型。

    s12 的 system prompt 会鼓励模型先用任务板规划，再为高风险/并行工作创建 worktree。
    """
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行任务和 worktree 工具。

    任务工具修改 .tasks/；worktree 工具修改 .worktrees/index.json、
    调用 git worktree，并把生命周期事件写入 .worktrees/events.jsonl。
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
# s12 的图结构和普通工具 agent 一样：
# START -> agent -> tools -> agent -> ...
#
# 复杂性不在图拓扑，而在工具副作用：
# 任务文件、git worktree、index 和事件日志共同构成任务隔离系统。
builder.add_edge(START, "agent")
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=MemorySaver())


def run_once(query: str, thread_id: str = "default") -> str:
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100})
    content = final_state["messages"][-1].content
    return content if isinstance(content, str) else str(content)


if __name__ == "__main__":
    print(f"graph-s12 使用的仓库根目录：{REPO_ROOT}")
    if not WORKTREES.git_available:
        print("注意：当前不在 git 仓库中，worktree_* 工具会返回错误。")
    thread_id = "cli-session"
    while True:
        try:
            query = input("\033[36mgraph-s12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        print(run_once(query, thread_id))
        print()
