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

# 当前脚本启动目录作为所有 agent 的工作区。
# lead 和 teammate 执行 shell、读写文件时都以这个目录为根。
WORKDIR = Path.cwd()

# 团队元数据保存在 .team/，消息 inbox 保存在 .team/inbox/。
# 这样 teammate 的名字、角色、状态和通信队列都不会混入模型上下文。
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# 模型名优先读取 OPENAI_MODEL，其次兼容其他示例使用的 MODEL_ID。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")

# 这是 lead agent 的系统提示。teammate 会在自己的线程里使用单独的系统提示。
SYSTEM = f"你是位于 {WORKDIR} 的团队负责人。请按需启动 teammate，并通过 inbox 与他们通信。"

# 文件消息总线允许的消息类型。
# s09 主要用 message/broadcast；后面的示例会继续扩展 shutdown 和 plan approval 协议。
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"}


class MessageBus:
    """基于 JSONL 文件的 append-only inbox。"""

    def __init__(self, inbox_dir: Path):
        # 每个 agent 都有一个独立 inbox 文件，例如 lead.jsonl、reviewer.jsonl。
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict | None = None) -> str:
        """向指定 agent 的 inbox 追加一条 JSONL 消息。"""
        # 限制消息类型，避免模型随意制造后续协议无法识别的 type。
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        # timestamp 使用 time.time()，方便之后按写入顺序或时间排序排查。
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            # extra 预留给更复杂协议，例如 approval id、任务 id 等扩展字段。
            msg.update(extra)

        # JSONL append-only 的好处是简单、跨线程可见，也便于人工查看。
        with (self.dir / f"{to}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"已向 {to} 发送 {msg_type}"

    def read_inbox(self, name: str) -> list:
        """读取并清空某个 agent 的 inbox。"""
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return []

        # 读取所有非空行并按 JSON 解析；每一行是一条独立消息。
        messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

        # 读完立即清空，形成“消息只消费一次”的队列语义。
        path.write_text("", encoding="utf-8")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        """向所有 teammate 发送广播消息。"""
        count = 0
        for name in teammates:
            # 不给发送者自己广播，避免产生无意义的自消息。
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"已广播给 {count} 个 teammate"


# 全局消息总线。lead 和所有 teammate 线程共享这个对象。
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
    # _run_* 是普通 Python 函数，供 lead tool 和 teammate 调度器共同复用。
    # @tool 包装函数只负责生成模型可见的工具 schema。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 同步执行命令；长任务在后续示例中会交给 background/task 系统。
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 输出截断，避免工具结果过大导致上下文膨胀。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int | None = None) -> str:
    """实际读取文件的内部实现。"""
    try:
        lines = _safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            # limit 适合模型先看文件头部，避免一次读入大文件。
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """实际写入文件的内部实现。"""
    try:
        fp = _safe_path(path)
        # 自动创建父目录，允许 agent 直接写入新路径。
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
            # 精确文本不存在时直接失败，避免模型猜测位置造成误改。
            return f"Error: Text not found in {path}"
        # 只替换第一处，降低一次工具调用的影响范围。
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("bash")
def bash(command: str) -> str:
    """执行 shell 命令。"""
    # 这是模型可见的工具入口；实际执行逻辑放在 _run_bash，方便复用。
    return _run_bash(command)


@tool("read_file")
def read_file(path: str, limit: int | None = None) -> str:
    """读取工作区文件。"""
    # lead 和 teammate 都可以使用这个工具 schema。
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
    # 这个函数只用于让 bind_tools 生成 send_message 的 schema。
    # teammate 运行时不能直接调用它，因为真正发送消息必须知道 sender 是谁。
    return "仅用于生成工具 schema"


@tool("read_inbox")
def read_inbox_schema() -> str:
    """读取并清空当前 agent 的 inbox。"""
    # 同理，这只是 teammate 可见的工具 schema。
    # 真正读取哪个 inbox 要由 TeammateManager._exec 根据当前 teammate 名字决定。
    return "仅用于生成工具 schema"


# teammate 可见的工具集合。
# send_message/read_inbox 在这里是 schema 占位，真正执行走 TeammateManager._exec。
TEAMMATE_TOOLS = [bash, read_file, write_file, edit_file, send_message_schema, read_inbox_schema]


# 基础 LLM 对象。lead 和每个 teammate 会分别 bind 不同的工具集合。
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
        # config.json 保存 team_name 和 members 列表；
        # 线程对象本身不能持久化，因此只放在内存中的 self.threads。
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads: dict[str, threading.Thread] = {}

        # spawn、状态更新、保存 config 都可能从不同线程发生，因此需要锁。
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

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """启动或复用一个命名 teammate，并把它放到后台线程中运行。"""
        with self._lock:
            member = self._find_member(name)
            if member:
                # 已存在的 teammate 只有在空闲或关闭状态时才能重新分配任务。
                # 如果它还在 working，重复启动可能导致同名线程争用同一个 inbox。
                if member["status"] not in ("idle", "shutdown"):
                    return f"Error: '{name}' is currently {member['status']}"
                member["status"] = "working"
                member["role"] = role
            else:
                # 第一次启动该 teammate 时，把它登记到持久配置中。
                member = {"name": name, "role": role, "status": "working"}
                self.config["members"].append(member)
            self._save_config()

        # 每个 teammate 拥有独立线程和独立消息上下文。
        # daemon=True 让主进程退出时不等待这些后台线程。
        thread = threading.Thread(target=self._teammate_loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"已启动 '{name}'（角色：{role}）"

    def _set_status(self, name: str, status: str) -> None:
        """线程安全地更新 teammate 状态。"""
        with self._lock:
            member = self._find_member(name)
            if member:
                member["status"] = status
                self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """执行 teammate 请求的工具调用。"""
        # 这里不直接调用 LangChain tool_obj.invoke，核心原因是 send_message/read_inbox
        # 需要知道当前 teammate 的身份 sender，而工具 schema 本身没有这个参数。
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            # sender 由线程上下文注入，避免模型伪造 from 字段。
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            # 每个 teammate 只能读取自己的 inbox。
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

        # 给该 teammate 绑定 teammate 工具 schema。
        # 注意：send_message/read_inbox 的真实执行仍然由 _exec 接管。
        model = llm.bind_tools(TEAMMATE_TOOLS)

        # teammate 的初始上下文只包含 lead 分配给它的任务 prompt。
        # 它不会继承 lead 的完整对话历史，避免上下文混杂。
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]

        for _ in range(50):
            # 每一轮模型调用前先读取自己的 inbox。
            # read_inbox 会清空 JSONL 文件，所以消息只会被消费一次。
            for msg in BUS.read_inbox(name):
                messages.append(HumanMessage(content=json.dumps(msg, ensure_ascii=False)))
            try:
                # 每轮都重新注入 teammate 身份和角色，降低长会话中身份漂移的风险。
                response = model.invoke([SystemMessage(content=sys_prompt)] + messages)
            except Exception:
                # 教学示例中简单退出线程；真实系统通常会记录异常并写回状态。
                break
            messages.append(response)
            if not isinstance(response, AIMessage) or not response.tool_calls:
                # 如果模型没有继续请求工具，说明当前 teammate 任务告一段落。
                break
            tool_messages = []
            for call in response.tool_calls:
                # teammate 的 send_message/read_inbox 需要知道 sender 身份，
                # 所以这里不直接调用 LangChain tool，而是走 self._exec(name, ...)。
                output = self._exec(name, call["name"], call.get("args", {}))
                print(f"  [{name}] {call['name']}: {str(output)[:120]}")

                # ToolMessage 带 tool_call_id，模型才能把结果对应回具体工具调用。
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
            messages.extend(tool_messages)

        # 线程自然结束后，如果 teammate 没被标记 shutdown，就回到 idle。
        if self._find_member(name) and self._find_member(name)["status"] != "shutdown":
            self._set_status(name, "idle")

    def list_all(self) -> str:
        """列出团队中所有 teammate 的角色和状态。"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for member in self.config["members"]:
            lines.append(f"  {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    def member_names(self) -> list[str]:
        """返回所有 teammate 名字，用于广播。"""
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器。lead 工具会通过它启动和查询 teammate。
TEAM = TeammateManager(TEAM_DIR)


@tool("spawn_teammate")
def spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动一个持久 teammate 线程。"""
    # lead 调用该工具后会立即得到结果，teammate 在后台线程中继续工作。
    return TEAM.spawn(name, role, prompt)


@tool("list_teammates")
def list_teammates() -> str:
    """列出所有 teammate。"""
    return TEAM.list_all()


@tool("send_message")
def lead_send_message(to: str, content: str, msg_type: str = "message") -> str:
    """lead 发送消息给 teammate。"""
    # lead 的 sender 固定为 "lead"，与 teammate 的 send_message 注入逻辑对称。
    return BUS.send("lead", to, content, msg_type)


@tool("read_inbox")
def lead_read_inbox() -> str:
    """读取并清空 lead inbox。"""
    # 这是 lead 主动读取 inbox 的工具；另外图里的 inbox 节点也会自动读取。
    return json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False)


@tool("broadcast")
def broadcast(content: str) -> str:
    """向所有 teammate 广播消息。"""
    # 广播目标来自当前 config.json 中登记的 teammate 列表。
    return BUS.broadcast("lead", content, TEAM.member_names())


# lead 可见的工具集合比 teammate 多：它可以 spawn/list/broadcast。
LEAD_TOOLS = [bash, read_file, write_file, edit_file, spawn_teammate, list_teammates, lead_send_message, lead_read_inbox, broadcast]

# 执行 lead 工具时按模型返回的工具名反查工具对象。
TOOL_BY_NAME = {t.name: t for t in LEAD_TOOLS}

# lead_llm 是绑定了 lead 工具集合的模型。
lead_llm = llm.bind_tools(LEAD_TOOLS)


class AgentState(TypedDict):
    """
    LangGraph 状态。

    lead 的消息历史由 LangGraph 管理；teammate 的消息历史保存在各自线程的局部变量中。
    add_messages 表示节点返回的新消息会追加到历史，而不是替换整个 messages。
    """
    messages: Annotated[list[BaseMessage], add_messages]


def inject_inbox_node(state: AgentState) -> dict:
    """
    每次 lead 调模型前，先把 lead inbox 注入上下文。

    inbox 是文件系统里的 JSONL 队列；读完后会清空。
    这让 teammate 可以异步写信给 lead，而 lead 在下一轮模型调用前看到这些消息。
    """
    inbox = BUS.read_inbox("lead")
    if not inbox:
        # 没有新消息时不修改状态。
        return {}

    # 用 <inbox> 标签包起来，让模型清楚这些内容来自异步队列，而不是用户新输入。
    return {"messages": [HumanMessage(content=f"<inbox>{json.dumps(inbox, indent=2, ensure_ascii=False)}</inbox>")]}


def agent_node(state: AgentState) -> dict:
    """调用 lead 模型，让它决定回答或请求工具。"""
    # 每轮都显式注入 SYSTEM；历史消息来自 LangGraph checkpointer。
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
            # LangChain tool.invoke 接收参数 dict，并按工具函数签名执行。
            output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
        except Exception as e:
            # 工具异常转成工具结果返回给模型，避免整个图执行崩溃。
            output = f"Error: {e}"
        print(f"> {name}:")
        print(str(output)[:200])

        # ToolMessage 必须带 tool_call_id，模型才能把结果和对应 tool_call 关联起来。
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

# tools_condition 是 LangGraph 预置条件：
# - lead 返回 tool_calls 时进入 tools 节点；
# - 没有 tool_calls 时走 __end__，结束本轮 run_once。
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})

# 工具执行完回到 inbox，而不是直接回 agent。
# 这样 teammate 在线程中刚写给 lead 的消息能在下一次模型调用前被注入。
builder.add_edge("tools", "inbox")

# MemorySaver 保存 lead 会话的消息历史。
# teammate 状态另存在 .team/config.json，消息另存在 .team/inbox/*.jsonl。
graph = builder.compile(checkpointer=MemorySaver())


def run_once(query: str, thread_id: str = "default") -> str:
    """执行一次 lead 用户输入，并返回本轮最终消息文本。"""
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100})
    content = final_state["messages"][-1].content
    return content if isinstance(content, str) else str(content)


if __name__ == "__main__":
    # CLI 模式固定使用同一个 thread_id，因此多轮输入会共享 lead 的 LangGraph 记忆。
    thread_id = "cli-session"
    while True:
        try:
            query = input("\033[36mgraph-s09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            # 调试命令：不走模型，直接查看当前团队配置。
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            # 调试命令：不走模型，直接读取并清空 lead inbox。
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue
        print(run_once(query, thread_id))
        print()
