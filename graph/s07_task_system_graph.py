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

# 当前示例默认以启动脚本的位置作为工作区。
# shell 命令、文件读写、任务目录都会围绕这个目录展开。
WORKDIR = Path.cwd()

# 所有任务会持久化到 .tasks/task_*.json。
# 这让任务列表不依赖模型上下文，即使对话历史被压缩也不会丢。
TASKS_DIR = WORKDIR / ".tasks"

# 模型名优先读取 OPENAI_MODEL，其次兼容其他示例常用的 MODEL_ID。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# 系统提示要求 agent 主动使用任务工具做规划和跟踪。
# s07 的重点不是“模型记住 todo”，而是“模型把 todo 写入持久任务系统”。
SYSTEM = f"你是位于 {WORKDIR} 的编程 agent。请使用任务工具规划、跟踪和更新工作。"


class TaskManager:
    """基于 JSON 文件的任务管理器，支持状态和 blockedBy 依赖。"""

    def __init__(self, tasks_dir: Path):
        # 任务目录可能不存在，启动时直接创建。
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)

        # _next_id 从已有任务文件中推导，避免脚本重启后从 1 开始覆盖旧任务。
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """扫描任务目录，找出当前最大的任务 id。"""
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                # 文件名格式是 task_123.json，stem 是 task_123。
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                # 忽略不符合命名格式的文件，避免一个脏文件让整个任务系统不可用。
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        """根据任务 id 生成对应 JSON 文件路径。"""
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        """读取并解析单个任务文件。"""
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, task: dict) -> None:
        """把任务对象写回 JSON 文件。"""
        # ensure_ascii=False 让中文任务标题和描述在文件中保持可读。
        self._path(task["id"]).write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        """创建一个 pending 状态的新任务，并返回 JSON 文本。"""
        task = {
            # id 是单调递增的整数，便于模型和用户引用。
            "id": self._next_id,
            "subject": subject,
            "description": description,

            # 状态约定：
            # pending 表示尚未开始，in_progress 表示正在处理，completed 表示完成。
            "status": "pending",

            # blockedBy 保存“当前任务依赖哪些任务完成”的 id 列表。
            # 例如 blockedBy=[2, 3] 表示 2 和 3 完成前，该任务不应开始。
            "blockedBy": [],

            # owner 预留给后续多 agent/多人协作示例使用；s07 暂不分配负责人。
            "owner": "",
        }
        self._save(task)

        # 写入成功后再递增 id，避免保存失败时跳号。
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        """返回单个任务的完整 JSON。"""
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str | None = None,
        addBlockedBy: list[int] | None = None,
        removeBlockedBy: list[int] | None = None,
    ) -> str:
        """更新任务状态和依赖关系。"""
        task = self._load(task_id)

        if status:
            # 只允许有限状态，避免模型写出 done、doing 等不一致字符串。
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status

            # 当某个任务完成后，其他任务对它的依赖应自动解除。
            # 这让依赖图能随着任务完成向前推进。
            if status == "completed":
                self._clear_dependency(task_id)

        if addBlockedBy:
            # 用 set 去重，再排序，保证 JSON 中依赖列表稳定、易读。
            task["blockedBy"] = sorted(set(task["blockedBy"] + addBlockedBy))

        if removeBlockedBy:
            # 只移除指定依赖，不影响其他 blockedBy。
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in removeBlockedBy]

        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int) -> None:
        """任务完成后，从其他任务的 blockedBy 中移除它。"""
        # 这里扫描所有任务文件，找到依赖 completed_id 的任务并更新。
        # 数据量很小时这种实现最简单；真实系统可以改成索引或数据库。
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text(encoding="utf-8"))
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """返回适合模型和 CLI 阅读的任务列表摘要。"""
        tasks = []

        # 按任务 id 排序，保证每次列出的顺序稳定。
        files = sorted(self.dir.glob("task_*.json"), key=lambda f: int(f.stem.split("_")[1]))
        for f in files:
            tasks.append(json.loads(f.read_text(encoding="utf-8")))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            # 用简单标记压缩状态信息，方便模型快速扫描。
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)


# 全局任务管理器。
# 它的内存里只保存 next_id，任务内容本身都在 .tasks/ 文件中。
TASKS = TaskManager(TASKS_DIR)


def safe_path(p: str) -> Path:
    """限制文件工具只能访问当前工作区。"""
    # resolve 会消除 .. 等路径片段，然后用 is_relative_to 阻止越权访问。
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool("bash")
def run_bash(command: str) -> str:
    """执行阻塞式 shell 命令。"""
    # 这是同步工具：图会等待命令完成后才继续调用模型。
    # 因此 timeout 设置为 120 秒，避免一次工具调用长期卡住。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # shell=True 让模型可以执行复合 shell 命令；同时也意味着必须做基本危险命令拦截。
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 输出截断，防止超长日志占满后续模型上下文。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """读取工作区文件。"""
    try:
        # safe_path 保证只能读取工作区内的文件。
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            # limit 让模型可以只看文件前 N 行，节省上下文。
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """写入工作区文件。"""
    try:
        fp = safe_path(path)
        # 自动创建父目录，方便模型写入新文件。
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
            # 精确文本不存在时直接失败，避免模型猜测位置造成误改。
            return f"Error: Text not found in {path}"
        # 只替换第一处，降低一次工具调用影响范围。
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("task_create")
def task_create(subject: str, description: str = "") -> str:
    """创建持久任务。"""
    # 返回完整 JSON，方便模型马上拿到 id 并继续添加依赖或更新状态。
    return TASKS.create(subject, description)


@tool("task_update")
def task_update(
    task_id: int,
    status: str | None = None,
    addBlockedBy: list[int] | None = None,
    removeBlockedBy: list[int] | None = None,
) -> str:
    """更新任务状态或依赖。"""
    # addBlockedBy/removeBlockedBy 使用 camelCase，是为了模拟很多工具 schema/API 的字段命名风格。
    return TASKS.update(task_id, status, addBlockedBy, removeBlockedBy)


@tool("task_list")
def task_list() -> str:
    """列出所有任务。"""
    # 列表输出是摘要格式，不是完整 JSON；适合规划时快速扫描。
    return TASKS.list_all()


@tool("task_get")
def task_get(task_id: int) -> str:
    """查看单个任务详情。"""
    # 查看单个任务时返回完整 JSON，包含 description、blockedBy、owner 等字段。
    return TASKS.get(task_id)


# 所有可供模型调用的工具。
# LangChain 会根据 @tool 包装的函数名、参数类型和 docstring 生成工具 schema。
TOOLS = [run_bash, run_read, run_write, run_edit, task_create, task_update, task_list, task_get]

# 执行工具时用模型返回的 tool name 反查具体工具对象。
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

# 绑定工具后的模型可以返回 tool_calls。
# 如果没有工具调用，则它的 AIMessage 就是本轮最终回答。
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """调用模型。"""
    # 每轮都显式加 SystemMessage；历史消息由 LangGraph state/checkpointer 提供。
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])

    # 返回的 messages 会通过 AgentState 中的 add_messages 追加到历史消息列表。
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行工具调用并生成 ToolMessage。

    task_create / task_update / task_list / task_get 都会访问 .tasks/。
    因此即使对话被压缩或程序重启，任务文件仍能保留下来。
    """
    # 只有 AIMessage 才可能包含模型请求的 tool_calls。
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    results = []
    for call in last.tool_calls or []:
        name = call["name"]
        tool_obj = TOOL_BY_NAME.get(name)
        try:
            # LangChain tool.invoke 接收参数 dict，并按工具函数签名调用实际函数。
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            # 工具异常不让程序崩溃，而是转成结果交回模型，让模型有机会修正输入。
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])

        # ToolMessage 必须带 tool_call_id，模型才能把结果和刚才的工具调用对应起来。
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=name))
    return {"messages": results}


builder = StateGraph(AgentState)

# s07 的图很直接：
# START -> agent -> tools -> agent -> ...
#
# agent 节点负责思考和决定是否调用工具；
# tools 节点负责执行工具并把结果回填给模型；
# 如果 agent 没有再请求工具，图就结束本轮调用。
builder.add_edge(START, "agent")
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

# tools_condition 是 LangGraph 预置条件：
# - 最后一条 AIMessage 有 tool_calls：进入 tools；
# - 没有 tool_calls：走 __end__，结束本轮。
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "agent")

# MemorySaver 负责保存同一个 thread_id 下的消息历史。
# 注意任务状态不靠 MemorySaver 保存，而是靠 .tasks/ JSON 文件持久化。
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
    # CLI 模式固定使用同一个 thread_id，因此多轮输入会共享 LangGraph 对话记忆。
    # 任务文件则始终保存在 .tasks/，不依赖这个 thread_id。
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
