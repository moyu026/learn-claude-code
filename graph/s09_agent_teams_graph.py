#!/usr/bin/env python3
"""
s09_agent_teams_graph.py - 使用 LangGraph 改写 Agent Teams


s09 的关键点：
- teammate 是持久命名 agent，不是 s04 那种一次性 subagent
- 每个 teammate 在自己的后台线程里运行
- 团队通信通过 .team/inbox/*.jsonl 文件完成
- lead agent 使用 LangGraph；teammate 线程内部也使用 ChatOpenAI tool calling
"""

import json
import os
import subprocess
import threading
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
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")
SYSTEM = f"你是位于 {WORKDIR} 的团队负责人。请按需启动 teammate，并通过 inbox 与他们通信。"
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}


class MessageBus:
    """基于 JSONL 文件的 append-only inbox。"""

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
    """发送消息到 teammate inbox。运行时会根据当前 agent 身份设置 sender。"""
    return "仅用于生成工具 schema"


@tool("read_inbox")
def read_inbox_schema() -> str:
    """读取并清空当前 agent 的 inbox。"""
    return "仅用于生成工具 schema"


TEAMMATE_TOOLS = [bash, read_file, write_file, edit_file, send_message_schema, read_inbox_schema]


llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)


class TeammateManager:
    """持久 teammate 管理器，状态保存到 .team/config.json。"""

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads: dict[str, threading.Thread] = {}
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

    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self._lock:
            member = self._find_member(name)
            if member:
                if member["status"] not in ("idle", "shutdown"):
                    return f"Error: '{name}' is currently {member['status']}"
                member["status"] = "working"
                member["role"] = role
            else:
                member = {"name": name, "role": role, "status": "working"}
                self.config["members"].append(member)
            self._save_config()

        thread = threading.Thread(target=self._teammate_loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"已启动 '{name}'（角色：{role}）"

    def _set_status(self, name: str, status: str) -> None:
        with self._lock:
            member = self._find_member(name)
            if member:
                member["status"] = status
                self._save_config()

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
        return f"Unknown tool: {tool_name}"

    def _teammate_loop(self, name: str, role: str, prompt: str) -> None:
        # 每个 teammate 都运行在独立后台线程中。
        #
        # 与 s04 subagent 的区别：
        # - s04 子 agent 执行完就销毁，只返回摘要
        # - s09 teammate 会保留在 .team/config.json 中，并在完成当前任务后进入 idle
        # - 其他成员可以继续通过 inbox 给它发消息
        #
        # 这里的 messages 是 teammate 自己的上下文，不会和 lead 的 messages 混在一起。
        sys_prompt = (
            f"你是 teammate '{name}'，角色是 {role}，当前工作目录是 {WORKDIR}。"
            "请使用 send_message 与其他成员沟通，并完成分配给你的任务。"
        )
        model = llm.bind_tools(TEAMMATE_TOOLS)
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]

        for _ in range(50):
            # 每一轮模型调用前先读取自己的 inbox。
            # read_inbox 会清空 JSONL 文件，所以消息只会被消费一次。
            for msg in BUS.read_inbox(name):
                messages.append(HumanMessage(content=json.dumps(msg, ensure_ascii=False)))
            try:
                response = model.invoke([SystemMessage(content=sys_prompt)] + messages)
            except Exception:
                break
            messages.append(response)
            if not isinstance(response, AIMessage) or not response.tool_calls:
                break
            tool_messages = []
            for call in response.tool_calls:
                # teammate 的 send_message/read_inbox 需要知道 sender 身份，
                # 所以这里不直接调用 LangChain tool，而是走 self._exec(name, ...)。
                output = self._exec(name, call["name"], call.get("args", {}))
                print(f"  [{name}] {call['name']}: {str(output)[:120]}")
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
            messages.extend(tool_messages)

        if self._find_member(name) and self._find_member(name)["status"] != "shutdown":
            self._set_status(name, "idle")

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for member in self.config["members"]:
            lines.append(f"  {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    def member_names(self) -> list[str]:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


@tool("spawn_teammate")
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动一个持久 teammate 线程。"""
    return TEAM.spawn(name, role, prompt)


@tool("list_teammates")
def list_teammates() -> str:
    """列出所有 teammate。"""
    return TEAM.list_all()


@tool("send_message")
def lead_send_message(to: str, content: str, msg_type: str = "message") -> str:
    """lead 发送消息给 teammate。"""
    return BUS.send("lead", to, content, msg_type)


@tool("read_inbox")
def lead_read_inbox() -> str:
    """读取并清空 lead inbox。"""
    return json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False)


@tool("broadcast")
def broadcast(content: str) -> str:
    """向所有 teammate 广播消息。"""
    return BUS.broadcast("lead", content, TEAM.member_names())


LEAD_TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates, lead_send_message, lead_read_inbox, broadcast]
TOOL_BY_NAME = {t.name: t for t in LEAD_TOOLS}
lead_llm = llm.bind_tools(LEAD_TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def inject_inbox_node(state: AgentState) -> dict:
    """
    每次 lead 调模型前，先把 lead inbox 注入上下文。

    inbox 是文件系统里的 JSONL 队列；读完后会清空。
    这让 teammate 可以异步写信给 lead，而 lead 在下一轮模型调用前看到这些消息。
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
    执行 lead 请求的工具。

    spawn_teammate 会启动后台线程；send_message/read_inbox/broadcast 会操作文件 inbox。
    工具结果统一包装成 ToolMessage，交回 lead 模型继续推理。
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
# lead 图的结构是：
# START -> inbox -> agent -> tools -> inbox -> ...
#
# inbox 节点负责把异步文件消息转成 HumanMessage；
# agent 节点负责调用模型；
# tools 节点负责执行 lead 工具。
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
            query = input("\033[36mgraph-s09 >> \033[0m")
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
