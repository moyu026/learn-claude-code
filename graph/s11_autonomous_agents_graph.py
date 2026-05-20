#!/usr/bin/env python3
"""
s11_autonomous_agents_graph.py - 使用 LangGraph 改写 Autonomous Agents

s11 在 s10 团队协议基础上增加自治能力：
- teammate 完成一轮工作后进入 idle
- idle 期间轮询 inbox
- 如果没有消息，则扫描 .tasks/ 中未认领、未阻塞的 pending task
- 找到任务后自动 claim，并恢复工作
"""

import json
import os
import subprocess
import threading
import time
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
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
SYSTEM = f"你是位于 {WORKDIR} 的团队负责人。teammate 具备自治能力，会在空闲时自己寻找任务。"
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}
shutdown_requests: dict[str, dict] = {}
plan_requests: dict[str, dict] = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


class MessageBus:
    """JSONL inbox 消息总线。"""

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict | None = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with (self.dir / f"{to}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"已向 {to} 发送 {msg_type}"

    def read_inbox(self, name: str) -> list:
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        path.write_text("", encoding="utf-8")
        return items

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"已广播给 {count} 个 teammate"


BUS = MessageBus(INBOX_DIR)


def scan_unclaimed_tasks() -> list[dict]:
    """扫描可自动认领的任务：pending、无 owner、无 blockedBy。"""
    TASKS_DIR.mkdir(exist_ok=True)
    tasks = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text(encoding="utf-8"))
        if task.get("status") == "pending" and not task.get("owner") and not task.get("blockedBy"):
            tasks.append(task)
    return tasks


def claim_task(task_id: int, owner: str) -> str:
    """原子化认领任务，避免多个 teammate 同时拿同一任务。"""
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text(encoding="utf-8"))
        if task.get("owner"):
            return f"Error: Task {task_id} has already been claimed by {task.get('owner')}"
        if task.get("status") != "pending":
            return f"Error: Task {task_id} cannot be claimed because its status is '{task.get('status')}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Claimed task #{task_id} for {owner}"


def make_identity_text(name: str, role: str, team_name: str) -> str:
    """长会话或压缩后重新注入 teammate 身份。"""
    return f"<identity>你是 '{name}'，角色是 {role}，团队是 {team_name}。请继续你的工作。</identity>"


def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = _safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("bash")
def bash(command: str) -> str:
    """执行 shell 命令。"""
    return _run_bash(command)


@tool("read_file")
def read_file(path: str, limit: int | None = None) -> str:
    """读取文件。"""
    return _run_read(path, limit)


@tool("write_file")
def write_file(path: str, content: str) -> str:
    """写入文件。"""
    return _run_write(path, content)


@tool("edit_file")
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """编辑文件。"""
    return _run_edit(path, old_text, new_text)


@tool("send_message")
def send_message_schema(to: str, content: str, msg_type: str = "message") -> str:
    """发送消息。"""
    return "仅用于生成工具 schema"


@tool("read_inbox")
def read_inbox_schema() -> str:
    """读取 inbox。"""
    return "仅用于生成工具 schema"


@tool("shutdown_response")
def shutdown_response_schema(request_id: str, approve: bool, reason: str = "") -> str:
    """回复关闭请求。"""
    return "仅用于生成工具 schema"


@tool("plan_approval")
def plan_approval_schema(plan: str) -> str:
    """提交计划审批。"""
    return "仅用于生成工具 schema"


@tool("idle")
def idle_schema() -> str:
    """进入 idle 轮询阶段。"""
    return "仅用于生成工具 schema"


@tool("claim_task")
def claim_task_schema(task_id: int) -> str:
    """认领任务。"""
    return "仅用于生成工具 schema"


TEAMMATE_TOOLS = [bash, read_file, write_file, edit_file, send_message_schema, read_inbox_schema, shutdown_response_schema, plan_approval_schema, idle_schema, claim_task_schema]

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)


class TeammateManager:
    """自治 teammate 管理器。"""

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
        self._lock = threading.Lock()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {"team_name": "default", "members": []}

    def _save_config(self) -> None:
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find_member(self, name: str) -> dict | None:
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def _set_status(self, name: str, status: str) -> None:
        with self._lock:
            member = self._find_member(name)
            if member:
                member["status"] = status
                self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self._lock:
            member = self._find_member(name)
            if member:
                if member["status"] not in ("idle", "shutdown"):
                    return f"Error: '{name}' is currently {member['status']}"
                member["status"] = "working"
                member["role"] = role
            else:
                self.config["members"].append({"name": name, "role": role, "status": "working"})
            self._save_config()
        thread = threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"已启动 '{name}'（角色：{role}）"

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response", {"request_id": req_id, "approve": args["approve"]})
            return f"关闭请求已{'批准' if args['approve'] else '拒绝'}"
        if tool_name == "plan_approval":
            req_id = str(uuid.uuid4())[:8]
            plan_text = args.get("plan", "")
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(sender, "lead", plan_text, "plan_approval_response", {"request_id": req_id, "plan": plan_text})
            return f"计划已提交（request_id={req_id}），等待审批。"
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        if tool_name == "idle":
            return "Entering idle phase. Will poll for new tasks."
        return f"Unknown tool: {tool_name}"

    def _loop(self, name: str, role: str, prompt: str) -> None:
        # teammate 生命周期分为两个阶段：
        #
        # 1. WORK 阶段：
        #    正常调用模型和工具，直到模型没有工具调用，或显式调用 idle。
        #
        # 2. IDLE 阶段：
        #    不调用模型，只轻量轮询 inbox 和 .tasks/。
        #    如果收到消息或找到可认领任务，就回到 WORK。
        #    如果超时仍没有工作，就把自己标记为 shutdown。
        team_name = self.config["team_name"]
        sys_prompt = (
            f"你是 teammate '{name}'，角色是 {role}，团队是 {team_name}，当前工作目录是 {WORKDIR}。"
            "当你暂时没有更多工作时，请调用 idle；进入 idle 后系统会帮你轮询消息和未认领任务。"
        )
        model = llm.bind_tools(TEAMMATE_TOOLS)
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]

        while True:
            for _ in range(50):
                # WORK 阶段每轮都先读 inbox。
                # shutdown_request 是特殊控制消息：收到后直接退出线程。
                for msg in BUS.read_inbox(name):
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append(HumanMessage(content=json.dumps(msg, ensure_ascii=False)))
                try:
                    response = model.invoke([SystemMessage(content=sys_prompt)] + messages)
                except Exception:
                    self._set_status(name, "idle")
                    return
                messages.append(response)
                if not isinstance(response, AIMessage) or not response.tool_calls:
                    break
                tool_messages = []
                idle_requested = False
                for call in response.tool_calls:
                    if call["name"] == "idle":
                        idle_requested = True
                    output = self._exec(name, call["name"], call.get("args", {}))
                    print(f"  [{name}] {call['name']}: {str(output)[:120]}")
                    tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
                messages.extend(tool_messages)
                if idle_requested:
                    break

            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                # IDLE 阶段不消耗 LLM token，只做轮询。
                # 这是 s11 的核心：agent 不需要用户再次提醒，就能发现新任务。
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append(HumanMessage(content=json.dumps(msg, ensure_ascii=False)))
                    resume = True
                    break
                tasks = scan_unclaimed_tasks()
                if tasks:
                    # 只认领 pending、无 owner、无 blockedBy 的任务。
                    # claim_task 内部有锁，避免多个 teammate 同时认领同一个任务。
                    task = tasks[0]
                    result = claim_task(task["id"], name)
                    if result.startswith("Error:"):
                        continue
                    messages.insert(0, HumanMessage(content=make_identity_text(name, role, team_name)))
                    messages.append(HumanMessage(content=f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"))
                    resume = True
                    break
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        return "\n".join([f"Team: {self.config['team_name']}"] + [f"  {m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]])

    def member_names(self) -> list[str]:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": req_id})
    return f"关闭请求 {req_id} 已发送给 '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response", {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"来自 '{req['from']}' 的计划已标记为 {req['status']}"


@tool("spawn_teammate")
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动自治 teammate。"""
    return TEAM.spawn(name, role, prompt)


@tool("list_teammates")
def list_teammates() -> str:
    """列出 teammate。"""
    return TEAM.list_all()


@tool("send_message")
def lead_send_message(to: str, content: str, msg_type: str = "message") -> str:
    """lead 发送消息。"""
    return BUS.send("lead", to, content, msg_type)


@tool("read_inbox")
def lead_read_inbox() -> str:
    """读取 lead inbox。"""
    return json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False)


@tool("broadcast")
def broadcast(content: str) -> str:
    """广播给所有 teammate。"""
    return BUS.broadcast("lead", content, TEAM.member_names())


@tool("shutdown_request")
def shutdown_request(teammate: str) -> str:
    """请求 teammate 关闭。"""
    return handle_shutdown_request(teammate)


@tool("shutdown_response")
def shutdown_status(request_id: str) -> str:
    """查询 shutdown request。"""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}), ensure_ascii=False)


@tool("plan_approval")
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """审批计划。"""
    return handle_plan_review(request_id, approve, feedback)


@tool("lead_claim_task")
def lead_claim_task(task_id: int) -> str:
    """lead 认领任务。"""
    return claim_task(task_id, "lead")


LEAD_TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates, lead_send_message, lead_read_inbox, broadcast, shutdown_request, shutdown_status, plan_approval, lead_claim_task]
TOOL_BY_NAME = {t.name: t for t in LEAD_TOOLS}
lead_llm = llm.bind_tools(LEAD_TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def inject_inbox_node(state: AgentState) -> dict:
    """
    把 lead inbox 注入上下文。

    自治 teammate 可能在后台提交计划、回复关闭请求或发送普通消息；
    lead 每轮模型调用前都先读取这些消息。
    """
    inbox = BUS.read_inbox("lead")
    if not inbox:
        return {}
    return {"messages": [HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2, ensure_ascii=False)}</inbox>")]}


def agent_node(state: AgentState) -> dict:
    response = lead_llm.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行 lead 工具。

    s11 的 lead 工具除了团队通信和协议工具外，还可以直接 claim task；
    teammate 则会在 idle 阶段自动扫描并认领任务。
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
# lead 仍然是一个标准 LangGraph tool-calling agent；
# 自治逻辑主要在 teammate 后台线程中。
builder.add_edge(START, "inbox")
builder.add_node("inbox", inject_inbox_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge("inbox", "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "inbox")
graph = builder.compile(checkpointer=MemorySaver())


def run_once(query: str, thread_id: str = "default") -> str:
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100})
    content = final_state["messages"][-1].content
    return content if isinstance(content, str) else str(content)


if __name__ == "__main__":
    thread_id = "cli-session"
    while True:
        try:
            query = input("\033[36mgraph-s11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                task = json.loads(f.read_text(encoding="utf-8"))
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task.get("status"), "[?]")
                owner = f" @{task['owner']}" if task.get("owner") else ""
                print(f"  {marker} #{task['id']}: {task['subject']}{owner}")
            continue
        print(run_once(query, thread_id))
        print()
