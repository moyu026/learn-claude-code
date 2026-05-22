#!/usr/bin/env python3
"""
s10_team_protocols_graph.py - 使用 LangGraph 改写 Team Protocols


s10 在 s09 团队 inbox 基础上增加两类协议：
- shutdown_request / shutdown_response：关闭请求用 request_id 关联响应
- plan_approval / plan_approval_response：计划审批也用 request_id 关联
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
# lead 和 teammate 执行命令、读写文件时都以这里为根目录。
WORKDIR = Path.cwd()

# 团队状态和通信文件都放在 .team/ 下：
# - config.json 保存 teammate 名称、角色和状态；
# - inbox/*.jsonl 保存 lead 与 teammate 之间的消息。
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# 模型名优先读取 OPENAI_MODEL，其次兼容仓库其他示例使用的 MODEL_ID。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# lead agent 的系统提示。s10 相比 s09 多了“关闭协议”和“计划审批协议”。
SYSTEM = f"你是位于 {WORKDIR} 的团队负责人。请使用关闭协议和计划审批协议管理 teammate。"

# MessageBus 只允许这些消息类型，避免模型写出后续协议无法识别的 type。
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}

# 关闭请求跟踪表：request_id -> {"target": teammate, "status": pending/approved/rejected}
shutdown_requests: dict[str, dict] = {}

# 计划审批跟踪表：request_id -> {"from": teammate, "plan": text, "status": pending/approved/rejected}
plan_requests: dict[str, dict] = {}

# lead 线程和 teammate 线程都会读写上面两个跟踪表，因此用同一把锁保护。
_tracker_lock = threading.Lock()


class MessageBus:
    """每个 teammate 一个 JSONL inbox。"""

    def __init__(self, inbox_dir: Path):
        # 每个 agent 对应一个 JSONL 文件，例如 lead.jsonl、builder.jsonl。
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict | None = None) -> str:
        """向指定 agent 的 inbox 追加一条消息。"""
        # 协议类消息依赖固定 type；未知 type 直接拒绝，方便调试。
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        # 基础字段每条消息都有；协议字段通过 extra 追加。
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            # extra 常用于 request_id、approve、feedback 等协议字段。
            msg.update(extra)

        # JSONL append-only 简单可靠，也便于人工查看消息流。
        with (self.dir / f"{to}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"已向 {to} 发送 {msg_type}"

    def read_inbox(self, name: str) -> list:
        """读取并清空某个 agent 的 inbox。"""
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []

        # 每一行是一条 JSON 消息；空行会被忽略。
        messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

        # 读完清空，形成“消息只交付一次”的队列语义。
        path.write_text("", encoding="utf-8")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        """向所有 teammate 广播普通消息。"""
        count = 0
        for name in teammates:
            # 不给发送者自己发，避免产生自循环消息。
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"已广播给 {count} 个 teammate"


# 全局消息总线。lead 和所有 teammate 线程共享它。
BUS = MessageBus(INBOX_DIR)


def _safe_path(p: str) -> Path:
    """把模型给出的路径限制在当前工作区内。"""
    # resolve 规整 .. 和符号链接，is_relative_to 防止越权访问工作区外文件。
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """实际执行 shell 命令的内部实现。"""
    # _run_* 是实现层，@tool 函数是模型可见的 schema 层。
    # 分开后 lead 工具和 teammate 自定义执行器都能复用同一份逻辑。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 同步运行命令；超时防止一次工具调用长期卡住整个 agent loop。
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 工具输出截断，避免超长日志挤占模型上下文。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int | None = None) -> str:
    """实际读取文件的内部实现。"""
    try:
        lines = _safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            # limit 允许模型先读文件头部，降低上下文消耗。
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """实际写入文件的内部实现。"""
    try:
        fp = _safe_path(path)
        # 自动创建父目录，便于 agent 写入新文件。
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
        # 只替换第一处，降低单次工具调用的影响范围。
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
    """读取工作区文件。"""
    return _run_read(path, limit)


@tool("write_file")
def write_file(path: str, content: str) -> str:
    """写入工作区文件。"""
    return _run_write(path, content)


@tool("edit_file")
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """替换文件中的第一处精确文本。"""
    return _run_edit(path, old_text, new_text)


@tool("send_message")
def send_message_schema(to: str, content: str, msg_type: str = "message") -> str:
    """给 teammate 发送消息。sender 由执行器按身份注入。"""
    # 仅用于给 teammate 生成 send_message 工具 schema。
    # 真正执行时必须由 TeammateManager._exec 注入 sender 身份。
    return "仅用于生成工具 schema"


@tool("read_inbox")
def read_inbox_schema() -> str:
    """读取当前 agent inbox。"""
    # 仅用于生成 schema；实际读取哪个 inbox 由当前 teammate 名字决定。
    return "仅用于生成工具 schema"


@tool("shutdown_response")
def shutdown_response_schema(request_id: str, approve: bool, reason: str = "") -> str:
    """回复 shutdown_request。"""
    # teammate 可见的协议工具 schema。
    # 真实执行会更新 shutdown_requests，并发 shutdown_response 给 lead。
    return "仅用于生成工具 schema"


@tool("plan_approval")
def plan_approval_schema(plan: str) -> str:
    """向 lead 提交计划审批。"""
    # teammate 可见的协议工具 schema。
    # 真实执行会创建 plan_requests，并把计划作为消息发给 lead。
    return "仅用于生成工具 schema"


# teammate 可见工具集合。
# 其中 send_message/read_inbox/shutdown_response/plan_approval 都是 schema 占位；
# 真正的副作用在 TeammateManager._exec 里按当前 teammate 身份执行。
TEAMMATE_TOOLS = [bash, read_file, write_file, edit_file, send_message_schema, read_inbox_schema, shutdown_response_schema, plan_approval_schema]

# 基础 LLM 对象。lead 和 teammate 会分别 bind 不同工具集合。
llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)


class TeammateManager:
    """带 shutdown / plan 协议的 teammate 管理器。"""

    def __init__(self, team_dir: Path):
        # config.json 持久化 teammate 名称、角色和状态；
        # thread 对象只能存在内存中，重启脚本后不会恢复旧线程。
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

        # spawn、状态更新、保存 config 都可能被不同线程触发，需要加锁。
        self._lock = threading.Lock()

    def _load_config(self) -> dict:
        """读取团队配置；不存在则返回默认空团队。"""
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
        """启动或复用一个命名 teammate。"""
        with self._lock:
            member = self._find_member(name)
            if member:
                # 已存在 teammate 只有在 idle/shutdown 时才能重新启动；
                # working 状态下重复启动会导致同名线程争抢同一个 inbox。
                if member["status"] not in ("idle", "shutdown"):
                    return f"Error: '{name}' is currently {member['status']}"
                member["status"] = "working"
                member["role"] = role
            else:
                # 首次启动时登记到持久团队配置中。
                self.config["members"].append({"name": name, "role": role, "status": "working"})
            self._save_config()

        # 每个 teammate 拥有独立后台线程和独立消息上下文。
        thread = threading.Thread(target=self._teammate_loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"已启动 '{name}'（角色：{role}）"

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # teammate 工具执行器。
        #
        # 这里集中处理协议相关副作用：
        # - shutdown_response：更新 shutdown_requests，并把响应发给 lead
        # - plan_approval：创建 plan_requests，并把计划发给 lead
        #
        # request_id 是协议的关键，它让 lead 能把后续响应和原始请求对应起来。
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
            approve = bool(args["approve"])
            with _tracker_lock:
                if req_id in shutdown_requests:
                    # lead 发起的 pending 请求在这里被 teammate 明确批准或拒绝。
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"

            # 响应也写入 lead inbox，方便 lead 模型在下一轮看到理由和 approve 字段。
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response", {"request_id": req_id, "approve": approve})
            return f"关闭请求已{'批准' if approve else '拒绝'}"
        if tool_name == "plan_approval":
            # teammate 提交计划时，由 teammate 端创建 request_id。
            # lead 后续审批时会用这个 id 找到对应计划。
            req_id = str(uuid.uuid4())[:8]
            plan_text = args.get("plan", "")
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}

            # 这里复用 plan_approval_response 作为“计划提交给 lead”的消息类型。
            # 消息里同时带 request_id 和 plan，lead 收到后可以调用 plan_approval 审批。
            BUS.send(sender, "lead", plan_text, "plan_approval_response", {"request_id": req_id, "plan": plan_text})
            return f"计划已提交（request_id={req_id}），等待 lead 审批。"
        return f"Unknown tool: {tool_name}"

    def _teammate_loop(self, name: str, role: str, prompt: str) -> None:
        # teammate 线程内部是一个普通 tool-calling loop。
        # 它不是 LangGraph 图，但使用同一套 ChatOpenAI.bind_tools 协议；
        # lead 侧才是 LangGraph，负责用户交互和团队总控。
        sys_prompt = (
            f"你是 teammate '{name}'，角色是 {role}，当前工作目录是 {WORKDIR}。"
            "开始重大工作前，请使用 plan_approval 提交计划。"
            "收到 shutdown_request 时，必须使用 shutdown_response 回复是否同意关闭。"
        )

        # 每个 teammate 绑定 teammate 工具 schema。
        # 协议工具的真实执行仍然经过 _exec，以便注入 sender 和更新全局跟踪表。
        model = llm.bind_tools(TEAMMATE_TOOLS)

        # teammate 初始上下文只包含 lead 分配的 prompt，不继承 lead 的完整会话。
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]

        # should_exit 由 shutdown_response 的 approve 决定。
        # 如果 teammate 批准关闭，当前 loop 会在本轮工具结果回填后退出。
        should_exit = False
        for _ in range(50):
            # 协议消息通过 inbox 到达 teammate。
            # 例如 lead 发送 shutdown_request 后，teammate 下一轮会读到该消息，
            # 并被系统提示要求用 shutdown_response 明确回复。
            for msg in BUS.read_inbox(name):
                messages.append(HumanMessage(content=json.dumps(msg, ensure_ascii=False)))
            if should_exit:
                break
            try:
                # 每轮都注入 teammate 身份、角色和协议要求，减少长会话身份漂移。
                response = model.invoke([SystemMessage(content=sys_prompt)] + messages)
            except Exception:
                # 教学示例中直接退出；生产系统通常会记录错误并通知 lead。
                break
            messages.append(response)
            if not isinstance(response, AIMessage) or not response.tool_calls:
                # 没有工具调用，说明 teammate 当前任务告一段落。
                break
            tool_messages = []
            for call in response.tool_calls:
                output = self._exec(name, call["name"], call.get("args", {}))
                print(f"  [{name}] {call['name']}: {str(output)[:120]}")
                if call["name"] == "shutdown_response" and call.get("args", {}).get("approve"):
                    # 先把 shutdown_response 的 ToolMessage 交回模型，再退出 loop。
                    should_exit = True
                # ToolMessage 需要 tool_call_id，模型才能对应到具体工具调用。
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
            messages.extend(tool_messages)

        # 如果批准关闭，最终状态是 shutdown；否则回到 idle，等待后续任务。
        self._set_status(name, "shutdown" if should_exit else "idle")

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
    # lead 发起关闭请求时，先创建 request_id，并把状态记录为 pending。
    # teammate 是否同意关闭，要等它通过 shutdown_response 回复后才知道。
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}

    # request_id 放进消息里，teammate 回复时必须带回同一个 id。
    BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": req_id})
    return f"关闭请求 {req_id} 已发送给 '{teammate}'（状态：pending）"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    # lead 审批 teammate 的计划。
    # request_id 来自 teammate 提交 plan_approval 时创建的 plan_requests。
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        # 审批结果写入跟踪表，后续可由调试或扩展示例查询。
        req["status"] = "approved" if approve else "rejected"

    # 审批结果通过 inbox 发回提交计划的 teammate。
    # teammate 下一轮读 inbox 时会看到 approve 和 feedback。
    BUS.send("lead", req["from"], feedback, "plan_approval_response", {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"来自 '{req['from']}' 的计划已标记为 {req['status']}"


@tool("spawn_teammate")
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动持久 teammate。"""
    # lead 工具：启动后立即返回，teammate 在后台线程中继续运行。
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
    # 这是手动读取工具；图里的 inbox 节点也会在每次模型调用前自动读取。
    return json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False)


@tool("broadcast")
def broadcast(content: str) -> str:
    """广播给所有 teammate。"""
    # 广播目标来自当前 config.json 中登记的 teammate。
    return BUS.broadcast("lead", content, TEAM.member_names())


@tool("shutdown_request")
def shutdown_request(teammate: str) -> str:
    """请求 teammate 优雅关闭，返回 request_id。"""
    # 发起请求不会立刻关闭 teammate，只会进入 pending，等待 teammate 回复。
    return handle_shutdown_request(teammate)


@tool("shutdown_response")
def shutdown_status(request_id: str) -> str:
    """查询 shutdown request 状态。"""
    # 这个工具让 lead 可以按 request_id 查询关闭请求是否已批准或拒绝。
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}), ensure_ascii=False)


@tool("plan_approval")
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """审批 teammate 提交的计划。"""
    # approve=True 表示允许执行计划；False 表示拒绝，并可通过 feedback 给出原因。
    return handle_plan_review(request_id, approve, feedback)


# lead 可见的工具集合：比 teammate 多了 spawn/list/broadcast，以及协议管理工具。
LEAD_TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates, lead_send_message, lead_read_inbox, broadcast, shutdown_request, shutdown_status, plan_approval]

# 执行 lead 工具时按模型返回的工具名反查工具对象。
TOOL_BY_NAME = {t.name: t for t in LEAD_TOOLS}

# lead 使用绑定了 lead 工具集合的模型。
lead_llm = llm.bind_tools(LEAD_TOOLS)


class AgentState(TypedDict):
    """
    LangGraph 状态。

    lead 的消息历史由 LangGraph 保存；teammate 的消息历史在各自线程局部变量中。
    协议请求状态额外保存在 shutdown_requests / plan_requests 两个内存表中。
    """
    messages: Annotated[list[BaseMessage], add_messages]


def inject_inbox_node(state: AgentState) -> dict:
    """
    把 lead inbox 中的协议消息注入上下文。

    s10 的 shutdown_response 和 plan_approval_response 都依靠 request_id 关联；
    这些响应先写入 lead inbox，再由本节点送进模型上下文。
    """
    inbox = BUS.read_inbox("lead")
    if not inbox:
        # 没有新协议消息时不修改图状态。
        return {}

    # 用标签包裹 inbox 内容，帮助模型区分“异步协议消息”和“用户输入”。
    return {"messages": [HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2, ensure_ascii=False)}</inbox>")]}


def agent_node(state: AgentState) -> dict:
    """调用 lead 模型，让它决定直接回答或继续调用工具。"""
    # 每轮都显式注入 SYSTEM；历史消息来自 LangGraph checkpointer。
    response = lead_llm.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行 lead 工具。

    shutdown_request 会创建 request_id 并记录到 shutdown_requests；
    plan_approval 会根据 request_id 更新 plan_requests，并把审批结果发回 teammate。
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

        # ToolMessage 必须带 tool_call_id，模型才能把结果和对应调用关联起来。
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=name))
    return {"messages": results}


builder = StateGraph(AgentState)
# lead 图在每次模型调用前都会先读 inbox。
# 这样 teammate 的协议响应不会丢失，也不会要求 lead 主动轮询文件。
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
# 这样 teammate 刚写入的协议响应可以在下一次模型调用前注入。
builder.add_edge("tools", "inbox")

# MemorySaver 保存 lead 的对话历史。
# 注意 shutdown_requests/plan_requests 是内存状态，重启脚本后会丢失；
# teammate 名单保存在 .team/config.json，inbox 消息保存在 .team/inbox/*.jsonl。
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
            query = input("\033[36mgraph-s10 >> \033[0m")
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
        print(run_once(query, thread_id))
        print()
