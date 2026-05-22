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

# 当前脚本启动目录作为所有 agent 的工作区。
# shell 命令、文件读写、团队配置和任务文件都围绕这个目录展开。
WORKDIR = Path.cwd()

# 团队状态和通信目录：
# - .team/config.json 保存 teammate 名字、角色和状态；
# - .team/inbox/*.jsonl 保存 lead 与 teammate 的异步消息。
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# s11 引入自治任务扫描，任务文件来自 s07 的 .tasks/task_*.json 设计。
TASKS_DIR = WORKDIR / ".tasks"

# 模型名优先读取 OPENAI_MODEL，其次兼容其他示例使用的 MODEL_ID。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# teammate 进入 idle 后，每隔 POLL_INTERVAL 秒检查 inbox 和可认领任务。
POLL_INTERVAL = 5

# idle 超过该时间仍没有新消息或可认领任务，就把 teammate 标记为 shutdown。
IDLE_TIMEOUT = 60

# lead 的系统提示。自治行为主要在 teammate 线程里实现，lead 只负责团队总控。
SYSTEM = f"你是位于 {WORKDIR} 的团队负责人。teammate 具备自治能力，会在空闲时自己寻找任务。"

# inbox 允许的消息类型。s11 延续 s10 的协议类型。
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}

# 关闭请求跟踪表：request_id -> {"target": teammate, "status": pending/approved/rejected}
shutdown_requests: dict[str, dict] = {}

# 计划审批跟踪表：request_id -> {"from": teammate, "plan": text, "status": pending/approved/rejected}
plan_requests: dict[str, dict] = {}

# 协议跟踪表会被 lead 线程和 teammate 线程共同访问，需要加锁。
_tracker_lock = threading.Lock()

# 任务认领锁：防止多个 idle teammate 同时扫描到同一个 pending 任务并重复认领。
_claim_lock = threading.Lock()


class MessageBus:
    """JSONL inbox 消息总线。"""

    def __init__(self, inbox_dir: Path):
        # 每个 agent 一个 inbox 文件，例如 lead.jsonl、builder.jsonl。
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict | None = None) -> str:
        """向指定 agent 的 inbox 追加一条 JSONL 消息。"""
        # 固定消息类型，避免模型写出协议层不认识的 type。
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        # 基础字段每条消息都有；协议字段通过 extra 扩展。
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            # extra 常用于 request_id、approve、feedback、plan 等字段。
            msg.update(extra)

        # JSONL append-only 简单、可人工查看，并且跨线程可见。
        with (self.dir / f"{to}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"已向 {to} 发送 {msg_type}"

    def read_inbox(self, name: str) -> list:
        """读取并清空某个 agent 的 inbox。"""
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []

        # 每行是一条 JSON 消息；空行会被忽略。
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

        # 读完清空，形成“只消费一次”的队列语义。
        path.write_text("", encoding="utf-8")
        return items

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        """向所有 teammate 广播消息。"""
        count = 0
        for name in teammates:
            # 不给发送者自己发，避免无意义的自消息。
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"已广播给 {count} 个 teammate"


# 全局消息总线，lead 和所有 teammate 线程共享。
BUS = MessageBus(INBOX_DIR)


def scan_unclaimed_tasks() -> list[dict]:
    """扫描可自动认领的任务：pending、无 owner、无 blockedBy。"""
    # idle teammate 会周期性调用它，寻找自己可以主动接手的工作。
    TASKS_DIR.mkdir(exist_ok=True)
    tasks = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text(encoding="utf-8"))
        # 只有未开始、无人认领、没有依赖阻塞的任务才适合自动认领。
        if task.get("status") == "pending" and not task.get("owner") and not task.get("blockedBy"):
            tasks.append(task)
    return tasks


def claim_task(task_id: int, owner: str) -> str:
    """原子化认领任务，避免多个 teammate 同时拿同一任务。"""
    with _claim_lock:
        # 整个读-检查-写过程必须在同一把锁内完成，否则会出现竞态：
        # 两个 teammate 同时看到 owner 为空，然后都写入自己的名字。
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

        # 认领成功后直接切到 in_progress，让其他 teammate 不再扫描到它。
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Claimed task #{task_id} for {owner}"


def make_identity_text(name: str, role: str, team_name: str) -> str:
    """长会话或压缩后重新注入 teammate 身份。"""
    # idle 很久后重新进入 WORK 时，把身份插回 messages 开头，降低角色漂移风险。
    return f"<identity>你是 '{name}'，角色是 {role}，团队是 {team_name}。请继续你的工作。</identity>"


def _safe_path(p: str) -> Path:
    """把模型给出的路径限制在当前工作区内。"""
    # resolve 规整 .. 和符号链接，is_relative_to 防止访问工作区外文件。
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """实际执行 shell 命令的内部实现。"""
    # _run_* 是实现层，@tool 函数是模型可见 schema 层。
    # teammate 的自定义执行器和 lead 工具都复用这些实现。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 同步执行命令，并设置超时防止工具调用长期卡住。
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 输出截断，避免工具结果过大撑满模型上下文。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int | None = None) -> str:
    """实际读取文件的内部实现。"""
    try:
        lines = _safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            # limit 允许模型先读取文件前几行，节省上下文。
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """实际写入文件的内部实现。"""
    try:
        fp = _safe_path(path)
        # 自动创建父目录，方便 agent 写入新文件。
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """实际编辑文件的内部实现，只替换第一处精确匹配。"""
    try:
        fp = _safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            # 精确文本找不到时直接失败，避免模型猜测位置造成误改。
            return f"Error: Text not found in {path}"
        # 只替换第一处，降低一次工具调用的影响范围。
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("bash")
def bash(command: str) -> str:
    """执行 shell 命令。"""
    # 模型可见的工具入口，实际逻辑委托给 _run_bash。
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
    # teammate 工具 schema 占位。
    # 真实执行需要由 TeammateManager._exec 注入 sender 身份。
    return "仅用于生成工具 schema"


@tool("read_inbox")
def read_inbox_schema() -> str:
    """读取 inbox。"""
    # schema 占位；实际读取哪个 inbox 由当前 teammate 名字决定。
    return "仅用于生成工具 schema"


@tool("shutdown_response")
def shutdown_response_schema(request_id: str, approve: bool, reason: str = "") -> str:
    """回复关闭请求。"""
    # schema 占位；真实执行会更新 shutdown_requests 并通知 lead。
    return "仅用于生成工具 schema"


@tool("plan_approval")
def plan_approval_schema(plan: str) -> str:
    """提交计划审批。"""
    # schema 占位；真实执行会创建 plan_requests 并通知 lead。
    return "仅用于生成工具 schema"


@tool("idle")
def idle_schema() -> str:
    """进入 idle 轮询阶段。"""
    # teammate 主动调用 idle 表示当前 WORK 阶段结束，系统可以进入低成本轮询。
    return "仅用于生成工具 schema"


@tool("claim_task")
def claim_task_schema(task_id: int) -> str:
    """认领任务。"""
    # schema 占位；真实执行会调用 claim_task(task_id, 当前 teammate 名字)。
    return "仅用于生成工具 schema"


# teammate 可见工具集合。
# send_message/read_inbox/shutdown_response/plan_approval/idle/claim_task 都由 _exec 接管真实执行。
TEAMMATE_TOOLS = [bash, read_file, write_file, edit_file, send_message_schema, read_inbox_schema, shutdown_response_schema, plan_approval_schema, idle_schema, claim_task_schema]

# 基础 LLM 对象。lead 和 teammate 会分别 bind 不同工具集合。
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
        # config.json 持久化 teammate 名称、角色和状态；
        # thread 对象只存在内存中，重启脚本后不会恢复旧线程。
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

        # spawn、状态更新、保存 config 可能来自不同线程，需要加锁。
        self._lock = threading.Lock()

    def _load_config(self) -> dict:
        """读取团队配置；不存在时返回默认空团队。"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {"team_name": "default", "members": []}

    def _save_config(self) -> None:
        """把团队配置写回 .team/config.json。"""
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find_member(self, name: str) -> dict | None:
        """按名字查找 teammate 配置。"""
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def _set_status(self, name: str, status: str) -> None:
        """线程安全地更新 teammate 状态。"""
        with self._lock:
            member = self._find_member(name)
            if member:
                member["status"] = status
                self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """启动或复用一个自治 teammate。"""
        with self._lock:
            member = self._find_member(name)
            if member:
                # 已存在 teammate 只有在 idle/shutdown 时才能重新分配任务。
                # working 状态下重复启动会导致同名线程争用同一个 inbox。
                if member["status"] not in ("idle", "shutdown"):
                    return f"Error: '{name}' is currently {member['status']}"
                member["status"] = "working"
                member["role"] = role
            else:
                # 首次启动该 teammate 时登记到持久团队配置中。
                self.config["members"].append({"name": name, "role": role, "status": "working"})
            self._save_config()

        # 每个 teammate 在独立后台线程中运行自己的 WORK/IDLE 生命周期。
        thread = threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"已启动 '{name}'（角色：{role}）"

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """执行 teammate 请求的工具调用。"""
        # teammate 的工具有些需要注入 sender，有些会更新协议跟踪表；
        # 因此不直接调用 LangChain tool.invoke，而是在这里统一调度。
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            # sender 由执行器注入，避免模型伪造 from 字段。
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            # teammate 只能读取自己的 inbox。
            return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    # lead 发起的关闭请求在这里被 teammate 明确批准或拒绝。
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"

            # 响应写入 lead inbox，lead 下一轮会看到 request_id 和 approve。
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response", {"request_id": req_id, "approve": args["approve"]})
            return f"关闭请求已{'批准' if args['approve'] else '拒绝'}"
        if tool_name == "plan_approval":
            # teammate 提交计划时创建 request_id，lead 后续用它审批。
            req_id = str(uuid.uuid4())[:8]
            plan_text = args.get("plan", "")
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}

            # 计划提交给 lead inbox，消息中带 request_id 和 plan 内容。
            BUS.send(sender, "lead", plan_text, "plan_approval_response", {"request_id": req_id, "plan": plan_text})
            return f"计划已提交（request_id={req_id}），等待审批。"
        if tool_name == "claim_task":
            # teammate 认领任务时，owner 强制使用 sender。
            return claim_task(args["task_id"], sender)
        if tool_name == "idle":
            # idle 是控制信号，不做外部副作用；_loop 会据此进入 IDLE 阶段。
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

        # 给 teammate 绑定 teammate 工具 schema。
        # 工具真实执行仍由 _exec 处理，以便注入身份和协议副作用。
        model = llm.bind_tools(TEAMMATE_TOOLS)

        # teammate 初始上下文只包含 lead 分配的 prompt，不继承 lead 的完整会话。
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
                    # 每轮都注入 teammate 身份和 idle 规则，降低长会话中的角色漂移。
                    response = model.invoke([SystemMessage(content=sys_prompt)] + messages)
                except Exception:
                    # 模型调用异常时不继续占用 working 状态，先回到 idle。
                    self._set_status(name, "idle")
                    return
                messages.append(response)
                if not isinstance(response, AIMessage) or not response.tool_calls:
                    # 没有工具调用，说明当前 WORK 阶段暂时结束。
                    break
                tool_messages = []
                idle_requested = False
                for call in response.tool_calls:
                    if call["name"] == "idle":
                        # idle 工具是模型主动声明“暂时没有更多工作”的信号。
                        idle_requested = True
                    output = self._exec(name, call["name"], call.get("args", {}))
                    print(f"  [{name}] {call['name']}: {str(output)[:120]}")

                    # ToolMessage 必须带 tool_call_id，模型才能对应到具体调用。
                    tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
                messages.extend(tool_messages)
                if idle_requested:
                    # 本轮工具结果已写回 messages，接下来进入 IDLE 轮询阶段。
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

                    # 自动认领任务后，在上下文开头重新注入身份，防止长时间 idle 后身份漂移。
                    messages.insert(0, HumanMessage(content=make_identity_text(name, role, team_name)))

                    # 把自动认领到的任务作为新输入追加给 teammate，让它回到 WORK 阶段处理。
                    messages.append(HumanMessage(content=f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"))
                    resume = True
                    break
            if not resume:
                # idle 超时仍没有任何工作，关闭该 teammate 线程。
                self._set_status(name, "shutdown")
                return

            # 找到消息或任务后恢复工作状态，下一轮 while 会重新进入 WORK。
            self._set_status(name, "working")

    def list_all(self) -> str:
        """列出所有 teammate 的角色和状态。"""
        if not self.config["members"]:
            return "No teammates."
        return "\n".join([f"Team: {self.config['team_name']}"] + [f"  {m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]])

    def member_names(self) -> list[str]:
        """返回所有 teammate 名字，用于广播。"""
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器，供 lead 工具调用。
TEAM = TeammateManager(TEAM_DIR)


def handle_shutdown_request(teammate: str) -> str:
    """lead 发起关闭请求，返回 request_id。"""
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}

    # request_id 放入消息，teammate 回复时会带回同一个 id。
    BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": req_id})
    return f"关闭请求 {req_id} 已发送给 '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """lead 审批 teammate 提交的计划。"""
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        # 审批状态写回内存跟踪表，便于后续查询或调试。
        req["status"] = "approved" if approve else "rejected"

    # 审批结果通过 inbox 发回提交计划的 teammate。
    BUS.send("lead", req["from"], feedback, "plan_approval_response", {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"来自 '{req['from']}' 的计划已标记为 {req['status']}"


@tool("spawn_teammate")
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动自治 teammate。"""
    # lead 工具：启动后立即返回，teammate 在后台线程中自治运行。
    return TEAM.spawn(name, role, prompt)


@tool("list_teammates")
def list_teammates() -> str:
    """列出 teammate。"""
    return TEAM.list_all()


@tool("send_message")
def lead_send_message(to: str, content: str, msg_type: str = "message") -> str:
    """lead 发送消息。"""
    # lead 的 sender 固定为 "lead"。
    return BUS.send("lead", to, content, msg_type)


@tool("read_inbox")
def lead_read_inbox() -> str:
    """读取 lead inbox。"""
    # 手动读取工具；图里的 inbox 节点也会在每次模型调用前自动读取。
    return json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False)


@tool("broadcast")
def broadcast(content: str) -> str:
    """广播给所有 teammate。"""
    # 广播目标来自当前 .team/config.json 中登记的 teammate。
    return BUS.broadcast("lead", content, TEAM.member_names())


@tool("shutdown_request")
def shutdown_request(teammate: str) -> str:
    """请求 teammate 关闭。"""
    # 发起请求后状态为 pending，等待 teammate 回复或直接处理 shutdown_request。
    return handle_shutdown_request(teammate)


@tool("shutdown_response")
def shutdown_status(request_id: str) -> str:
    """查询 shutdown request。"""
    # 按 request_id 查询关闭请求状态。
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}), ensure_ascii=False)


@tool("plan_approval")
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """审批计划。"""
    # approve=True 表示允许执行计划；False 表示拒绝，并可通过 feedback 说明原因。
    return handle_plan_review(request_id, approve, feedback)


@tool("lead_claim_task")
def lead_claim_task(task_id: int) -> str:
    """lead 认领任务。"""
    # lead 也可以直接认领任务，使用同一套 claim_task 原子检查逻辑。
    return claim_task(task_id, "lead")


# lead 可见工具集合：包括团队通信、协议工具，以及 lead_claim_task。
LEAD_TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates, lead_send_message, lead_read_inbox, broadcast, shutdown_request, shutdown_status, plan_approval, lead_claim_task]

# 执行 lead 工具时按模型返回的工具名反查工具对象。
TOOL_BY_NAME = {t.name: t for t in LEAD_TOOLS}

# lead 使用绑定了 lead 工具集合的模型。
lead_llm = llm.bind_tools(LEAD_TOOLS)


class AgentState(TypedDict):
    """
    LangGraph 状态。

    lead 的消息历史由 LangGraph 保存；
    teammate 的消息历史保存在各自后台线程的局部变量中。
    自治任务状态则落在 .tasks/task_*.json 文件里。
    """
    messages: Annotated[list[BaseMessage], add_messages]


def inject_inbox_node(state: AgentState) -> dict:
    """
    把 lead inbox 注入上下文。

    自治 teammate 可能在后台提交计划、回复关闭请求或发送普通消息；
    lead 每轮模型调用前都先读取这些消息。
    """
    inbox = BUS.read_inbox("lead")
    if not inbox:
        # 没有新消息时不修改图状态。
        return {}

    # 用标签包裹 inbox 内容，帮助模型区分异步消息和用户输入。
    return {"messages": [HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2, ensure_ascii=False)}</inbox>")]}


def agent_node(state: AgentState) -> dict:
    """调用 lead 模型，让它决定直接回答或继续调用工具。"""
    # 每轮都显式注入 SYSTEM；历史消息来自 LangGraph checkpointer。
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
            # LangChain tool.invoke 接收参数 dict，并按工具函数签名执行。
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            # 工具异常转成 ToolMessage 返回给模型，避免整轮图崩溃。
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])

        # ToolMessage 必须带 tool_call_id，模型才能关联到对应工具调用。
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

# tools_condition 是 LangGraph 预置条件：
# - 最后一条 AIMessage 有 tool_calls，则进入 tools；
# - 没有 tool_calls，则结束本轮 run_once。
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})

# 工具执行完回到 inbox，而不是直接回 agent。
# 这样 teammate 刚写入 lead inbox 的消息能在下一次模型调用前注入。
builder.add_edge("tools", "inbox")

# MemorySaver 保存 lead 的消息历史。
# teammate 状态保存在 .team/config.json，任务状态保存在 .tasks/*.json。
graph = builder.compile(checkpointer=MemorySaver())


def run_once(query: str, thread_id: str = "default") -> str:
    """执行一次 lead 用户输入，并返回本轮最终消息文本。"""
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100})
    content = final_state["messages"][-1].content
    return content if isinstance(content, str) else str(content)


if __name__ == "__main__":
    # CLI 模式固定使用同一个 thread_id，因此多轮输入共享 lead 的 LangGraph 记忆。
    thread_id = "cli-session"
    while True:
        try:
            query = input("\033[36mgraph-s11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            # 调试命令：直接查看团队状态，不经过模型。
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            # 调试命令：直接读取并清空 lead inbox，不经过模型。
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue
        if query.strip() == "/tasks":
            # 调试命令：直接查看 .tasks/ 中的任务摘要，不经过模型。
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                task = json.loads(f.read_text(encoding="utf-8"))
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task.get("status"), "[?]")
                owner = f" @{task['owner']}" if task.get("owner") else ""
                print(f"  {marker} #{task['id']}: {task['subject']}{owner}")
            continue
        print(run_once(query, thread_id))
        print()
