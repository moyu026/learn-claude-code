#!/usr/bin/env python3
"""
s06_context_compact_graph.py - 使用 LangGraph 改写 Context Compact 示例


s06 展示三层压缩：
- micro_compact：每轮模型调用前，压缩较旧的工具结果
- auto_compact：估算上下文超过阈值时，保存 transcript 并让模型总结
- compact 工具：模型主动请求压缩时，立即触发总结
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

load_dotenv(override=True)

WORKDIR = Path.cwd()
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")
SYSTEM = f"你是位于 {WORKDIR} 的编程 agent。请使用工具解决任务；需要压缩上下文时可以调用 compact。"

THRESHOLD = 50000
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}
TRANSCRIPT_DIR = WORKDIR / ".transcripts"


class AgentState(TypedDict):
    # s06 需要整体替换 messages，所以这里不使用 add_messages reducer。
    messages: list[BaseMessage]
    manual_compact: bool


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """粗略估算 token 数：约 4 个字符一个 token。"""
    return len(str(messages)) // 4


def micro_compact(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    压缩较旧的 ToolMessage。

    LangChain 版本里工具结果是 ToolMessage，而不是 Anthropic 的 tool_result dict。
    这里保留最近 KEEP_RECENT 个工具结果；较旧且内容较长的结果替换成占位文本。
    read_file 的结果默认保留，因为文件内容通常是后续推理依据。
    """
    tool_indexes = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]

    if len(tool_indexes) <= KEEP_RECENT:
        return messages

    for idx in tool_indexes[:-KEEP_RECENT]:
        msg = messages[idx]
        tool_name = getattr(msg, "name", "") or "unknown"

        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        if isinstance(msg.content, str) and len(msg.content) > 100:
            messages[idx] = ToolMessage(
                content=f"[Previous: used {tool_name}]",
                tool_call_id=msg.tool_call_id,
                name=tool_name,
            )

    return messages


llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)


def auto_compact(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    保存完整 transcript，然后让模型总结历史，并用摘要替换 messages。

    注意：这里用同一个 ChatOpenAI 对象，但不绑定工具，因为压缩任务只需要纯文本总结。
    """
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"

    with transcript_path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.model_dump(), default=str, ensure_ascii=False) + "\n")

    print(f"[transcript saved: {transcript_path}]")

    conversation_text = json.dumps(
        [m.model_dump() for m in messages],
        default=str,
        ensure_ascii=False,
    )[-80000:]

    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    "请总结下面的对话，供后续继续工作使用。必须包含："
                    "1）已经完成了什么；2）当前状态；3）做过的关键决策。"
                    "请简洁，但保留会影响后续执行的关键细节。\n\n"
                    + conversation_text
                )
            )
        ]
    )
    summary = response.content if isinstance(response.content, str) else str(response.content)
    summary = summary or "没有生成摘要。"

    return [
        HumanMessage(
            content=f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"
        )
    ]


def safe_path(p: str) -> Path:
    """把路径限制在当前工作区内，避免文件工具越界。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool("bash")
def run_bash(command: str) -> str:
    """在当前工作区执行 shell 命令。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except Exception as e:
        return f"Error: {e}"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """读取当前工作区内的文件。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """写入当前工作区内的文件。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """替换文件中第一次出现的精确文本。"""
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("compact")
def run_compact(focus: str | None = None) -> str:
    """请求立即压缩对话。focus 可说明摘要应重点保留什么。"""
    return "Manual compression requested."


TOOLS = [run_bash, run_read, run_write, run_edit, run_compact]
TOOL_BY_NAME = {t.name: t for t in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)


def prepare_node(state: AgentState) -> dict:
    """
    每次调用模型前先进入这个节点。

    它对应原版 s06 agent_loop 里 while True 顶部的两步：
    1. micro_compact：静默压缩旧工具结果
    2. auto_compact：如果上下文过大，就保存 transcript 并替换成摘要

    这样压缩逻辑是图的一部分，而不是散落在 CLI 循环里。
    """
    messages = micro_compact(list(state["messages"]))
    if estimate_tokens(messages) > THRESHOLD:
        print("[auto_compact triggered]")
        messages = auto_compact(messages)
    return {"messages": messages, "manual_compact": False}


def agent_node(state: AgentState) -> dict:
    """
    调用绑定工具后的模型。

    注意这里的 messages 没有使用 add_messages reducer，
    因为 s06 需要在压缩时整体替换历史；所以节点返回完整 messages。
    """
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": state["messages"] + [response], "manual_compact": state.get("manual_compact", False)}


def tools_node(state: AgentState) -> dict:
    """
    执行工具调用；如果模型调用 compact，则标记本轮结束后要压缩。

    compact 工具本身不直接总结，因为工具执行阶段还需要先把 compact 的
    ToolMessage 写回历史，保证模型工具调用协议完整；随后再执行 auto_compact。
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return state

    tool_messages = []
    manual_compact = False

    for call in last.tool_calls or []:
        name = call["name"]
        if name == "compact":
            manual_compact = True
            output = "Compressing..."
        else:
            tool_obj = TOOL_BY_NAME.get(name)
            try:
                output = tool_obj.invoke(call.get("args", {})) if tool_obj else f"Unknown tool: {name}"
            except Exception as e:
                output = f"Error: {e}"

        print(f"> {name}:")
        print(str(output)[:200])
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=name))

    messages = state["messages"] + tool_messages

    if manual_compact:
        print("[manual compact]")
        messages = auto_compact(messages)

    return {"messages": messages, "manual_compact": manual_compact}


def after_tools(state: AgentState) -> str:
    """手动 compact 后直接结束；普通工具调用后继续回到模型。"""
    return "__end__" if state.get("manual_compact") else "prepare"


builder = StateGraph(AgentState)
builder.add_edge(START, "prepare")
builder.add_node("prepare", prepare_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge("prepare", "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_conditional_edges("tools", after_tools, {"prepare": "prepare", "__end__": "__end__"})
graph = builder.compile()


def run_once(query: str, history: list[BaseMessage] | None = None) -> tuple[str, list[BaseMessage]]:
    """执行一次输入，返回最终文本和新历史。"""
    messages = list(history or []) + [HumanMessage(content=query)]
    final_state = graph.invoke({"messages": messages, "manual_compact": False}, config={"recursion_limit": 100})
    last = final_state["messages"][-1]
    text = last.content if isinstance(last.content, str) else str(last.content)
    return text, final_state["messages"]


def agent_loop(messages: list[BaseMessage]) -> None:
    """兼容原版 agent_loop(messages)。"""
    final_state = graph.invoke({"messages": messages, "manual_compact": False}, config={"recursion_limit": 100})
    messages[:] = final_state["messages"]


if __name__ == "__main__":
    history: list[BaseMessage] = []
    while True:
        try:
            query = input("\033[36mgraph-s06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        answer, history = run_once(query, history)
        print(answer)
        print()
