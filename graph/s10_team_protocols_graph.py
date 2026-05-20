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

WORKDIR = Path.cwd()
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")
SYSTEM = f"你是位于 {WORKDIR} 的团队负责人。请使用关闭协议和计划审批协议管理 teammate。"
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}
shutdown_requests: dict[str, dict] = {}
plan_requests: dict[str, dict] = {}
_tracker_lock = threading.Lock()


class MessageBus:
    """每个 teammate 一个 JSONL inbox。"""

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
        messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        path.write_text("", encoding="utf-8")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"已广播给 {count} 个 teammate"


BUS = MessageBus(INBOX_DIR)


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
    return "仅用于生成工具 schema"


@tool("read_inbox")
def read_inbox_schema() -> str:
    """读取当前 agent inbox。"""
    return "仅用于生成工具 schema"


@tool("shutdown_response")
def shutdown_response_schema(request_id: str, approve: bool, reason: str = "") -> str:
    """回复 shutdown_request。"""
    return "仅用于生成工具 schema"


@tool("plan_approval")
def plan_approval_schema(plan: str) -> str:
    """向 lead 提交计划审批。"""
    return "仅用于生成工具 schema"


TEAMMATE_TOOLS = [bash, read_file, write_file, edit_file, send_message_schema, read_inbox_schema, shutdown_response_schema, plan_approval_schema]

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
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = bool(args["approve"])
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            BUS.send(sender, "lead", args.get("reason", ""), "shutdown_response", {"request_id": req_id, "approve": approve})
            return f"关闭请求已{'批准' if approve else '拒绝'}"
        if tool_name == "plan_approval":
            req_id = str(uuid.uuid4())[:8]
            plan_text = args.get("plan", "")
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
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
        model = llm.bind_tools(TEAMMATE_TOOLS)
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]
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
                response = model.invoke([SystemMessage(content=sys_prompt)] + messages)
            except Exception:
                break
            messages.append(response)
            if not isinstance(response, AIMessage) or not response.tool_calls:
                break
            tool_messages = []
            for call in response.tool_calls:
                output = self._exec(name, call["name"], call.get("args", {}))
                print(f"  [{name}] {call['name']}: {str(output)[:120]}")
                if call["name"] == "shutdown_response" and call.get("args", {}).get("approve"):
                    should_exit = True
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
            messages.extend(tool_messages)
        self._set_status(name, "shutdown" if should_exit else "idle")

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        return "\n".join([f"Team: {self.config['team_name']}"] + [f"  {m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]])

    def member_names(self) -> list[str]:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


def handle_shutdown_request(teammate: str) -> str:
    # lead 发起关闭请求时，先创建 request_id，并把状态记录为 pending。
    # teammate 是否同意关闭，要等它通过 shutdown_response 回复后才知道。
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
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
        req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response", {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"来自 '{req['from']}' 的计划已标记为 {req['status']}"


@tool("spawn_teammate")
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动持久 teammate。"""
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
    """请求 teammate 优雅关闭，返回 request_id。"""
    return handle_shutdown_request(teammate)


@tool("shutdown_response")
def shutdown_status(request_id: str) -> str:
    """查询 shutdown request 状态。"""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}), ensure_ascii=False)


@tool("plan_approval")
def plan_approval(request_id: str, approve: bool, feedback: str = "") -> str:
    """审批 teammate 提交的计划。"""
    return handle_plan_review(request_id, approve, feedback)


LEAD_TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates, lead_send_message, lead_read_inbox, broadcast, shutdown_request, shutdown_status, plan_approval]
TOOL_BY_NAME = {t.name: t for t in LEAD_TOOLS}
lead_llm = llm.bind_tools(LEAD_TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def inject_inbox_node(state: AgentState) -> dict:
    """
    把 lead inbox 中的协议消息注入上下文。

    s10 的 shutdown_response 和 plan_approval_response 都依靠 request_id 关联；
    这些响应先写入 lead inbox，再由本节点送进模型上下文。
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
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])
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
            query = input("\033[36mgraph-s10 >> \033[0m")
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
        print(run_once(query, thread_id))
        print()
