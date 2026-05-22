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

# 当前脚本启动目录。普通文件工具和普通 bash 工具默认在这里执行。
WORKDIR = Path.cwd()

# 模型名优先读取 OPENAI_MODEL，其次兼容仓库其他示例使用的 MODEL_ID。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")


def detect_repo_root(cwd: Path) -> Path | None:
    """如果当前目录在 git 仓库内，返回仓库根目录。"""
    try:
        # worktree 必须以 git 仓库为基础；先通过 git rev-parse 找到真实仓库根。
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


# 如果当前目录不在 git 仓库里，就退化为 WORKDIR。
# 这种情况下任务工具仍可用，但 worktree_* 工具会因为 git_available=False 报错。
REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR

# s12 的系统提示鼓励模型把任务规划和执行隔离结合起来：
# 先在 .tasks/ 创建任务，再为风险较高或并行的任务创建独立 worktree。
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
        # 生命周期事件统一写入 .worktrees/events.jsonl。
        # 它是追加日志，不作为当前状态的唯一来源；当前状态主要在 index.json。
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, task: dict | None = None, worktree: dict | None = None, error: str | None = None) -> None:
        """追加一条生命周期事件。"""
        # task/worktree 只放必要快照，避免日志复制完整大对象。
        payload = {"event": event, "ts": time.time(), "task": task or {}, "worktree": worktree or {}}
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        """返回最近 N 条事件，方便 agent 或用户审计 worktree 生命周期。"""
        # 限制最多 200 条，避免一次工具调用把历史日志全部塞进上下文。
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        items = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except Exception:
                # 日志里如果混入坏行，不让整个查询失败。
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2, ensure_ascii=False)


class TaskManager:
    """持久任务板，并支持 worktree 绑定。"""

    def __init__(self, tasks_dir: Path):
        # .tasks 是任务控制面：记录“要做什么、谁负责、绑定哪个 worktree”。
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)

        # 从已有任务文件推导下一个 id，避免脚本重启后覆盖旧任务。
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """扫描 .tasks/task_*.json，找出当前最大任务 id。"""
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                # 忽略不符合 task_N.json 命名的文件。
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        """根据任务 id 生成任务文件路径。"""
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        """读取并解析单个任务。"""
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, task: dict) -> None:
        """写回任务 JSON，并刷新 updated_at。"""
        task["updated_at"] = time.time()
        # ensure_ascii=False 让中文任务标题/描述在文件中保持可读。
        self._path(task["id"]).write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        """创建一个 pending 任务。"""
        task = {
            # id 是单调递增的整数，方便模型和用户引用。
            "id": self._next_id,
            "subject": subject,
            "description": description,

            # pending -> in_progress -> completed 是本示例支持的状态流。
            "status": "pending",

            # owner 记录任务负责人；worktree 记录隔离执行通道名称。
            "owner": "",
            "worktree": "",

            # blockedBy 延续 s07 的依赖字段；s12 不展开依赖清理，但保留结构。
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        # 保存成功后再递增，避免写入失败造成跳号。
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        """返回单个任务完整 JSON。"""
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def exists(self, task_id: int) -> bool:
        """检查任务文件是否存在，供 worktree 创建前校验使用。"""
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str | None = None, owner: str | None = None) -> str:
        """更新任务状态或 owner。"""
        task = self._load(task_id)
        if status:
            # 限制状态枚举，避免模型写出 done/doing 等不一致状态。
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            # owner=None 表示不修改；owner="" 可以清空负责人。
            task["owner"] = owner
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        """把任务绑定到指定 worktree 名称。"""
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            # 一旦绑定执行通道，pending 任务自动进入 in_progress。
            task["status"] = "in_progress"
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def unbind_worktree(self, task_id: int) -> str:
        """清空任务上的 worktree 绑定。"""
        task = self._load(task_id)
        task["worktree"] = ""
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def list_all(self) -> str:
        """返回任务摘要列表，适合 CLI 和模型快速扫描。"""
        tasks = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(self.dir.glob("task_*.json"))]
        if not tasks:
            return "No tasks."
        lines = []
        for task in tasks:
            # 紧凑显示状态、owner 和 worktree 绑定关系。
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task["status"], "[?]")
            owner = f" owner={task['owner']}" if task.get("owner") else ""
            wt = f" wt={task['worktree']}" if task.get("worktree") else ""
            lines.append(f"{marker} #{task['id']}: {task['subject']}{owner}{wt}")
        return "\n".join(lines)


# 全局任务板和事件日志都放在仓库根目录下，而不是 WORKDIR。
# 这样即使脚本从仓库子目录启动，任务和 worktree 元数据仍集中在仓库根。
TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


class WorktreeManager:
    """管理 git worktree 和 .worktrees/index.json。"""

    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        # repo_root 是 git worktree 命令的执行根目录。
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events

        # .worktrees 是执行隔离面：里面放独立 worktree 目录、index 和事件日志。
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            # index.json 保存当前已知 worktree 的状态列表。
            # 它是当前状态视图，events.jsonl 是历史审计日志。
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2), encoding="utf-8")
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        """确认 repo_root 是否可用作 git 仓库。"""
        try:
            r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=self.repo_root, capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        """在 repo_root 中执行 git 命令，并在失败时抛出异常。"""
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(["git", *args], cwd=self.repo_root, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            # 把 stdout/stderr 合并成错误信息，便于模型直接看到 git 失败原因。
            raise RuntimeError((r.stdout + r.stderr).strip() or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        """读取 .worktrees/index.json。"""
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, data: dict) -> None:
        """保存 .worktrees/index.json。"""
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find(self, name: str) -> dict | None:
        """按 worktree 名称在 index 中查找记录。"""
        for wt in self._load_index().get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name: str) -> None:
        """限制 worktree 名称，避免路径穿越和难处理的 shell 字符。"""
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

        # before/after/failed 三段事件能完整描述创建尝试的生命周期。
        self.events.emit("worktree.create.before", task={"id": task_id} if task_id is not None else {}, worktree={"name": name, "base_ref": base_ref})
        try:
            # git worktree add -b 会从 base_ref 创建独立分支和独立工作目录。
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            # index 记录执行通道元数据，方便后续 list/status/run/remove。
            entry = {"name": name, "path": str(path), "branch": branch, "task_id": task_id, "status": "active", "created_at": time.time()}
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)
            if task_id is not None:
                # 绑定任务后，任务会自动从 pending 进入 in_progress。
                self.tasks.bind_worktree(task_id, name)
            self.events.emit("worktree.create.after", task={"id": task_id} if task_id is not None else {}, worktree=entry)
            return json.dumps(entry, indent=2, ensure_ascii=False)
        except Exception as e:
            self.events.emit("worktree.create.failed", task={"id": task_id} if task_id is not None else {}, worktree={"name": name}, error=str(e))
            raise

    def list_all(self) -> str:
        """列出 index 中记录的 worktree。"""
        items = self._load_index().get("worktrees", [])
        if not items:
            return "No worktrees in index."
        return "\n".join(f"[{wt.get('status', 'unknown')}] {wt['name']} -> {wt['path']} ({wt.get('branch', '-')})" for wt in items)

    def status(self, name: str) -> str:
        """在指定 worktree 目录里查看 git status。"""
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(["git", "status", "--short", "--branch"], cwd=path, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip() or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        """在指定 worktree 目录中执行 shell 命令。"""
        # 这是隔离执行的核心：命令 cwd 是 worktree path，而不是主 WORKDIR。
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
            # worktree_run 给更长超时，因为隔离通道通常用来跑测试或构建。
            r = subprocess.run(command, shell=True, cwd=path, capture_output=True, text=True, timeout=300)
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def keep(self, name: str) -> str:
        """保留 worktree，标记为 kept，不删除目录。"""
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
        # keep 表示这个隔离通道值得保留，可能用于后续人工检查或继续开发。
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
                # --force 用于删除有未提交改动等 git 默认拒绝删除的 worktree。
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)
            if complete_task and wt.get("task_id") is not None:
                # 删除执行通道时可选地把绑定任务标记为完成。
                self.tasks.update(wt["task_id"], status="completed")
                self.tasks.unbind_worktree(wt["task_id"])
                self.events.emit("task.completed", task={"id": wt["task_id"], "status": "completed"}, worktree={"name": name})
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    # 不从 index 物理删除，保留审计记录。
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)
            self.events.emit("worktree.remove.after", task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {}, worktree={"name": name, "status": "removed"})
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit("worktree.remove.failed", task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {}, worktree={"name": name}, error=str(e))
            raise


# 全局 worktree 管理器。所有 worktree_* 工具都通过它操作 git 和元数据。
WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


def safe_path(p: str) -> Path:
    """把普通文件工具的路径限制在当前 WORKDIR 内。"""
    # 注意这里使用 WORKDIR，而不是 REPO_ROOT。
    # 普通 read/write/edit 面向用户启动脚本所在工作区；worktree_run 才进入隔离目录。
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool("bash")
def bash(command: str) -> str:
    """在当前工作区执行阻塞命令。"""
    # 这是普通同步命令工具，cwd 是 WORKDIR。
    # 如果要在隔离执行通道里运行命令，应使用 worktree_run。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 普通 bash 超时较短，避免 agent 被一次工具调用卡住。
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 截断输出，防止长日志占满模型上下文。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


@tool("read_file")
def read_file(path: str, limit: int | None = None) -> str:
    """读取当前工作区文件。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            # limit 让模型可以先看文件头部，减少上下文占用。
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def write_file(path: str, content: str) -> str:
    """写入当前工作区文件。"""
    try:
        fp = safe_path(path)
        # 自动创建父目录，方便写入新文件。
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
            # 精确匹配失败时不做模糊编辑，避免误改相似代码。
            return f"Error: Text not found in {path}"
        # 只替换第一处，降低单次工具调用的影响范围。
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("task_create")
def task_create(subject: str, description: str = "") -> str:
    """创建任务。"""
    # 创建的任务落在 REPO_ROOT/.tasks，而不是当前对话消息里。
    return TASKS.create(subject, description)


@tool("task_list")
def task_list() -> str:
    """列出任务。"""
    # 返回摘要列表，适合模型快速判断当前任务状态。
    return TASKS.list_all()


@tool("task_get")
def task_get(task_id: int) -> str:
    """查看任务。"""
    # 返回完整任务 JSON，包括 owner、worktree、时间戳等字段。
    return TASKS.get(task_id)


@tool("task_update")
def task_update(task_id: int, status: str | None = None, owner: str | None = None) -> str:
    """更新任务状态或 owner。"""
    return TASKS.update(task_id, status, owner)


@tool("task_bind_worktree")
def task_bind_worktree(task_id: int, worktree: str, owner: str = "") -> str:
    """把任务绑定到 worktree。"""
    # 手动绑定只修改任务 JSON；它不会创建 git worktree。
    # 通常更推荐直接用 worktree_create(name, task_id) 同时创建和绑定。
    return TASKS.bind_worktree(task_id, worktree, owner)


@tool("worktree_create")
def worktree_create(name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
    """创建 git worktree。"""
    # 创建隔离执行通道，并可选绑定任务。
    # 该工具会同时触碰 git worktree、index.json、任务 JSON 和 events.jsonl。
    return WORKTREES.create(name, task_id, base_ref)


@tool("worktree_list")
def worktree_list() -> str:
    """列出 worktree index。"""
    # 读取 .worktrees/index.json 当前状态视图。
    return WORKTREES.list_all()


@tool("worktree_status")
def worktree_status(name: str) -> str:
    """查看 worktree git 状态。"""
    # 在隔离 worktree 目录中执行 git status，而不是主工作区。
    return WORKTREES.status(name)


@tool("worktree_run")
def worktree_run(name: str, command: str) -> str:
    """在指定 worktree 中运行命令。"""
    # 隔离执行命令的主要入口。cwd 是 .worktrees/<name>。
    return WORKTREES.run(name, command)


@tool("worktree_keep")
def worktree_keep(name: str) -> str:
    """标记 worktree 为 kept。"""
    # keep 只更新 index 和事件日志，不删除 worktree 目录。
    return WORKTREES.keep(name)


@tool("worktree_remove")
def worktree_remove(name: str, force: bool = False, complete_task: bool = False) -> str:
    """删除 worktree，可选完成绑定任务。"""
    # remove 调用 git worktree remove，并把 index 状态标记为 removed。
    # complete_task=True 时还会把绑定任务标记 completed 并解绑。
    return WORKTREES.remove(name, force, complete_task)


@tool("worktree_events")
def worktree_events(limit: int = 20) -> str:
    """查看最近 lifecycle events。"""
    # 查询 append-only 事件日志，适合排查创建、保留、删除失败等生命周期问题。
    return EVENTS.list_recent(limit)


# 模型可见的全部工具。
# s12 同时暴露普通文件/命令工具、任务工具和 worktree 隔离工具。
TOOLS = [
    bash, read_file, write_file, edit_file,
    task_create, task_list, task_get, task_update, task_bind_worktree,
    worktree_create, worktree_list, worktree_status, worktree_run, worktree_keep, worktree_remove, worktree_events,
]

# 执行工具调用时根据模型返回的工具名查找具体工具对象。
TOOL_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    """
    LangGraph 状态。

    对话消息由 LangGraph/MemorySaver 保存；
    任务、worktree index 和生命周期事件都持久化在文件系统中。
    """
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)

# 绑定工具后的模型可以返回 tool_calls；没有 tool_calls 时就是最终回答。
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """
    调用模型。

    s12 的 system prompt 会鼓励模型先用任务板规划，再为高风险/并行工作创建 worktree。
    """
    # 每轮都显式注入 SYSTEM；历史消息来自 LangGraph state/checkpointer。
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    # add_messages 会把返回的 AIMessage 追加进消息历史。
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行任务和 worktree 工具。

    任务工具修改 .tasks/；worktree 工具修改 .worktrees/index.json、
    调用 git worktree，并把生命周期事件写入 .worktrees/events.jsonl。
    """
    # 模型请求的工具调用挂在最后一条 AIMessage 上。
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    results = []
    for call in last.tool_calls or []:
        name = call["name"]
        tool_obj = TOOL_BY_NAME.get(name)
        try:
            # LangChain tool.invoke 接收参数 dict，并按工具函数签名执行。
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            # 工具异常转换成 ToolMessage 返回给模型，让模型能修正参数或解释失败。
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])
        # ToolMessage 必须带 tool_call_id，模型才能把结果对应回具体调用。
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

# tools_condition 是 LangGraph 预置条件：
# - 最后一条 AIMessage 有 tool_calls，则进入 tools；
# - 没有 tool_calls，则结束本轮 run_once。
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "agent")

# MemorySaver 只保存对话历史。
# s12 的关键状态另存在 .tasks、.worktrees/index.json 和 .worktrees/events.jsonl。
graph = builder.compile(checkpointer=MemorySaver())


def run_once(query: str, thread_id: str = "default") -> str:
    """执行一次用户输入，并返回本轮最终消息文本。"""
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100})
    content = final_state["messages"][-1].content
    return content if isinstance(content, str) else str(content)


if __name__ == "__main__":
    # 启动时打印实际使用的仓库根目录，方便确认 .tasks 和 .worktrees 会写到哪里。
    print(f"graph-s12 使用的仓库根目录：{REPO_ROOT}")
    if not WORKTREES.git_available:
        # 非 git 仓库中仍可使用任务工具，但 git worktree 相关工具会失败。
        print("注意：当前不在 git 仓库中，worktree_* 工具会返回错误。")

    # CLI 模式固定使用同一个 thread_id，因此多轮输入共享 LangGraph 对话记忆。
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
