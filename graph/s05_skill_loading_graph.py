#!/usr/bin/env python3
# 示例目标：用 LangGraph 表达按需加载技能的 agent loop。
"""
s05_skill_loading_graph.py - 使用 LangGraph 改写 Skills 示例

这是 agents/s05_skill_loading.py 的 LangGraph / langchain_openai 版本。

s05 的核心思想是：不要把所有专业知识一次性塞进 system prompt，
而是做两层注入。

第一层：便宜的技能摘要

    system prompt:
      Skills available:
        - pdf: Process PDF files...
        - code-review: Review code...

第二层：按需加载完整技能内容

    model 调用 load_skill("pdf")
      -> tool result:
         <skill name="pdf">
         完整 PDF 处理说明...
         </skill>

这样模型一开始知道有哪些技能，但只有在真正需要时才把完整技能正文加载进上下文。
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Annotated

import yaml
from dotenv import load_dotenv
from typing_extensions import TypedDict

# LangChain 消息类型。
#
# AIMessage：
#   模型回复。模型请求工具时，tool_calls 字段里有工具名、参数和调用 id。
#
# BaseMessage：
#   所有消息类型的基类，用于给 LangGraph state 做类型标注。
#
# HumanMessage：
#   用户消息。CLI 输入会包装成 HumanMessage。
#
# SystemMessage：
#   系统提示词。这里会包含技能摘要，也就是 s05 的第一层注入。
#
# ToolMessage：
#   工具执行结果。load_skill 返回的完整技能正文会通过 ToolMessage 交回模型。
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# @tool 把普通 Python 函数包装成 LangChain Tool。
# 包装后可以交给 ChatOpenAI.bind_tools(...)，让模型看到工具 schema。
from langchain_core.tools import tool

# graph 目录下的改写版本统一使用 OpenAI-compatible 模型入口。
from langchain_openai import ChatOpenAI

# StateGraph 定义有状态工作流；START 是图的起点。
from langgraph.graph import START, StateGraph

# add_messages 让节点返回的新消息追加到历史里，而不是覆盖原 messages。
from langgraph.graph.message import add_messages

# tools_condition 根据最后一条 AIMessage 是否包含 tool_calls 来决定是否进入工具节点。
from langgraph.prebuilt import tools_condition

# MemorySaver 让 CLI 中同一个 thread_id 可以保留多轮对话历史。
from langgraph.checkpoint.memory import MemorySaver


# ------------------------------------------------------------
# 1. 环境变量和模型配置
# ------------------------------------------------------------

load_dotenv(override=True)

# 当前工作目录。
WORKDIR = Path.cwd()

# 技能目录。原版 s05 约定技能放在 skills/<name>/SKILL.md。
SKILLS_DIR = WORKDIR / "skills"

# graph 版本使用 ChatOpenAI，因此优先读取 OPENAI_MODEL。
#
# MODEL_ID 是 agents/*.py 的 Anthropic SDK 版本使用的变量；
# OPENAI_MODEL 才是 graph/*.py 的 OpenAI-compatible 版本使用的变量。
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID", "gpt-4.1")


# ------------------------------------------------------------
# 2. SkillLoader：扫描并按需读取技能
# ------------------------------------------------------------

class SkillLoader:
    """
    加载 skills/<name>/SKILL.md 的轻量工具。

    每个 SKILL.md 可以包含 YAML frontmatter：

        ---
        name: pdf
        description: Process PDF files
        tags: documents
        ---
        技能正文...

    SkillLoader 会把 frontmatter 作为 meta，把后面的 Markdown 作为 body。
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict[str, object]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """
        扫描所有 SKILL.md。

        这里只在模块加载时扫描一次，保持示例简单。
        如果技能文件运行中会变化，可以增加 reload_skill / reload_all 工具。
        """
        if not self.skills_dir.exists():
            return

        for skill_file in sorted(self.skills_dir.rglob("SKILL.md")):
            text = skill_file.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = str(meta.get("name", skill_file.parent.name))
            self.skills[name] = {
                "meta": meta,
                "body": body,
                "path": str(skill_file),
            }

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """
        解析 SKILL.md 开头的 YAML frontmatter。

        如果没有 frontmatter，就返回空 meta，并把整个文件当作正文。
        如果 YAML 解析失败，也退化为空 meta，避免一个坏技能阻塞整个 agent。
        """
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)

        if not match:
            return {}, text

        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}

        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """
        第一层注入：只返回技能名和短描述。

        这段文本会进入 system prompt，让模型知道有哪些技能可用，
        但不会把完整技能正文提前放进上下文。
        """
        if not self.skills:
            return "(no skills available)"

        lines = []

        for name, skill in self.skills.items():
            meta = skill["meta"]
            desc = meta.get("description", "No description")
            tags = meta.get("tags", "")
            line = f"  - {name}: {desc}"

            if tags:
                line += f" [{tags}]"

            lines.append(line)

        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """
        第二层注入：按名称返回完整技能正文。

        这个方法由 load_skill 工具调用，结果会作为 ToolMessage 进入模型上下文。
        """
        skill = self.skills.get(name)

        if not skill:
            available = ", ".join(self.skills.keys())
            return f"Error: Unknown skill '{name}'. Available: {available}"

        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)


# ------------------------------------------------------------
# 3. 系统提示词：只注入技能摘要
# ------------------------------------------------------------

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


# ------------------------------------------------------------
# 4. 文件路径安全检查
# ------------------------------------------------------------

def safe_path(p: str) -> Path:
    """
    把模型传入的路径转换成工作区内的绝对路径。

    这样 read_file / write_file / edit_file 都不能逃逸到项目目录之外。
    """
    path = (WORKDIR / p).resolve()

    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")

    return path


# ------------------------------------------------------------
# 5. 工具实现
# ------------------------------------------------------------

@tool("bash")
def run_bash(command: str) -> str:
    """
    在当前工作区执行 shell 命令。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

    except Exception as e:
        return f"Error: {e}"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """
    读取当前工作区内的文件。

    参数：
    - path：相对于工作区的文件路径
    - limit：可选，只返回前 N 行
    """
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """
    写入当前工作区内的文件。

    父目录不存在时自动创建；文件已存在时整体覆盖。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"

    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    在工作区文件中替换第一次出现的精确文本。
    """
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")

        if old_text not in content:
            return f"Error: Text not found in {path}"

        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


@tool("load_skill")
def run_load_skill(name: str) -> str:
    """
    按名称加载完整技能说明。

    参数：
    - name：技能名，例如 pdf、code-review、agent-builder
    """
    return SKILL_LOADER.get_content(name)


TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
    run_load_skill,
]

TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}


# ------------------------------------------------------------
# 6. LangGraph State
# ------------------------------------------------------------

class AgentState(TypedDict):
    """
    LangGraph 节点之间共享的状态。

    messages 使用 add_messages reducer：
    每个节点只需要返回新增消息，LangGraph 会自动追加到历史后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------
# 7. 模型初始化
# ------------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
    temperature=0,
    max_tokens=8000,
)

# bind_tools 只把工具 schema 暴露给模型。
# 真正执行工具的是 tools_node。
llm_with_tools = llm.bind_tools(TOOLS)


# ------------------------------------------------------------
# 8. Graph 节点
# ------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    调用模型一次。

    每次调用都临时把 SYSTEM 放在最前面；
    state["messages"] 只保存用户、模型和工具之间的交互历史。
    """
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM)] + state["messages"]
    )

    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    执行最后一条 AIMessage 中的所有工具调用。

    对 s05 来说，最重要的工具是 load_skill：
    模型第一次只知道技能摘要，调用 load_skill 后才会收到完整技能正文。
    """
    last = state["messages"][-1]

    if not isinstance(last, AIMessage):
        return {}

    tool_messages = []

    for call in last.tool_calls or []:
        tool_name = call["name"]
        tool_args = call.get("args", {})
        tool_obj = TOOL_BY_NAME.get(tool_name)

        if tool_obj is None:
            output = f"Unknown tool: {tool_name}"
        else:
            try:
                output = tool_obj.invoke(tool_args)
            except Exception as e:
                output = f"Error: {e}"

        print(f"> {tool_name}:")
        print(str(output)[:200])

        tool_messages.append(
            ToolMessage(
                content=str(output)[:50000],
                tool_call_id=call["id"],
            )
        )

    return {"messages": tool_messages}


# ------------------------------------------------------------
# 9. 构建 LangGraph
# ------------------------------------------------------------

builder = StateGraph(AgentState)

# START -> agent
builder.add_edge(START, "agent")

# agent 节点负责调用模型。
builder.add_node("agent", agent_node)

# tools 节点负责执行工具，并把 ToolMessage 返回给模型。
builder.add_node("tools", tools_node)

# agent -> tools 或 agent -> END。
# 如果模型返回 tool_calls，就执行工具；否则图结束。
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "__end__",
    },
)

# 工具结果追加到 messages 后，再回到模型继续推理。
builder.add_edge("tools", "agent")


# ------------------------------------------------------------
# 10. 编译 Graph
# ------------------------------------------------------------

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


# ------------------------------------------------------------
# 11. 对外调用封装
# ------------------------------------------------------------

def message_text(message: BaseMessage) -> str:
    """
    把 LangChain message.content 转成普通字符串。

    大多数模型返回 str；有些 OpenAI-compatible 适配器可能返回 list 结构，
    因此这里做一点兼容处理。
    """
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def run_once(query: str, thread_id: str = "default") -> str:
    """
    执行一次用户输入，并返回最终模型文本。

    同一个 thread_id 会复用历史消息，所以 CLI 可以多轮对话。
    """
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
        },
    )

    return message_text(final_state["messages"][-1])


def agent_loop(messages: list[BaseMessage]) -> None:
    """
    兼容原版 s05 的 agent_loop(messages) 调用方式。

    外部可以自己维护 messages 列表；执行完成后这里会用最终 state 覆盖原列表。
    """
    final_state = graph.invoke(
        {"messages": messages},
        config={
            "configurable": {"thread_id": f"agent-loop-{id(messages)}-{len(messages)}"},
            "recursion_limit": 100,
        },
    )

    messages[:] = final_state["messages"]


# ------------------------------------------------------------
# 12. CLI 入口
# ------------------------------------------------------------

if __name__ == "__main__":
    thread_id = "cli-session"

    while True:
        try:
            query = input("\033[36mgraph-s05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        answer = run_once(query, thread_id=thread_id)
        print(answer)
        print()
