#!/usr/bin/env python3
# Harness: the agent loop expressed as a LangGraph StateGraph.
r"""
s01_agent_loop_langgraph.py - LangGraph Agent Loop

This keeps the original s01 behavior, but moves the control flow into
LangGraph:

    +-------+    tool_use     +---------+
    |  llm  | --------------> |  tools  |
    +---+---+                 +----+----+
        ^                          |
        |       tool_result        |
        +--------------------------+

The graph stops when the model response has no tool calls.
"""

import os
import subprocess
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ============================================================
# 1. 加载环境变量
# ============================================================
# 会从当前目录下的 .env 文件中读取环境变量。
#
# override=True 表示：
# 如果系统环境变量里已经有同名变量，也允许 .env 中的值覆盖它。
#
# 常见 .env 内容：
#
# OPENAI_API_KEY=xxx
# OPENAI_BASE_URL=https://xxx/v1
# OPENAI_MODEL=gpt-4.1
#
load_dotenv(override=True)


# ============================================================
# 2. 模型配置
# ============================================================
# 从环境变量中读取模型名。
# 如果没有配置 OPENAI_MODEL，则默认使用 gpt-4.1。
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")


# ============================================================
# 3. 系统提示词
# ============================================================
# os.getcwd() 会获取当前 Python 进程所在目录。
#
# 这个 SYSTEM 的作用是告诉模型：
# - 你是一个 coding agent
# - 当前工作目录在哪里
# - 可以使用 bash 工具解决任务
# - 不要只解释，要实际行动
#
# 例如模型可能会调用 bash：
#   ls
#   cat file.py
#   python test.py
#
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


# ============================================================
# 4. 定义 LangGraph 的状态结构
# ============================================================
# LangGraph 的核心是 StateGraph。
#
# 每个节点接收一个 state，返回一个新的 state。
#
# 这里的 AgentState 表示整个 Agent 运行过程中维护的状态。
#
# messages 是对话历史，里面会包含：
# - HumanMessage：用户消息
# - AIMessage：模型回复
# - ToolMessage：工具执行结果
#
# 这个 messages 会不断累积：
#
# 用户问题 -> LLM 回复 tool_call -> 工具执行结果 -> LLM 继续回复
#
class AgentState(TypedDict):
    messages: list[BaseMessage]


# ============================================================
# 5. 定义 bash 工具
# ============================================================
# @tool("bash") 会把这个 Python 函数包装成一个 LangChain Tool。
#
# 模型 bind_tools 后，就可以选择调用这个工具。
#
# 这个工具的输入参数是：
#   command: str
#
# 对模型来说，它看到的工具大概是：
#
# name: bash
# description: Run a shell command.
# args:
#   command: string
#
@tool("bash")
def run_bash(command: str) -> str:
    """
    Run a shell command.

    这个 docstring 很重要。
    它会作为工具描述提供给 LLM。
    模型会根据工具名、参数 schema、描述来判断什么时候调用工具。
    """

    # ------------------------------------------------------------
    # 5.1 简单的危险命令拦截
    # ------------------------------------------------------------
    # 这里做了一个非常基础的安全限制。
    #
    # 注意：
    # 这只是 demo 级别的安全拦截，不能作为真正沙箱。
    #
    # 工程实践中一般会使用：
    # - Docker 容器
    # - 权限隔离
    # - 文件系统白名单
    # - 命令白名单
    # - 超时限制
    # - 网络限制
    #
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # ------------------------------------------------------------
        # 5.2 执行 shell 命令
        # ------------------------------------------------------------
        # subprocess.run 用于执行命令。
        #
        # shell=True 表示通过系统 shell 执行命令。
        #
        # cwd=os.getcwd() 表示命令在当前目录执行。
        #
        # capture_output=True 表示捕获 stdout 和 stderr。
        #
        # text=True 表示输出转成字符串，而不是 bytes。
        #
        # timeout=120 表示最多执行 120 秒，防止命令卡死。
        #
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )

        # stdout 是正常输出。
        # stderr 是错误输出。
        # 这里把两者拼起来返回给模型。
        out = (r.stdout + r.stderr).strip()

        # ------------------------------------------------------------
        # 5.3 限制工具输出长度
        # ------------------------------------------------------------
        # 工具输出会重新塞回 LLM 上下文。
        #
        # 如果命令输出太长，会导致：
        # - token 太多
        # - 上下文污染
        # - 模型注意力分散
        # - 后续推理质量下降
        #
        # 所以这里最多返回 50000 个字符。
        #
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        # 命令超时
        return "Error: Timeout (120s)"

    except (FileNotFoundError, OSError) as e:
        # 例如命令不存在、路径异常等
        return f"Error: {e}"


# ============================================================
# 6. 工具列表
# ============================================================
# 一个 Agent 可以绑定多个工具。
#
# 这里目前只有 bash 一个工具。
#
TOOLS = [run_bash]


# ============================================================
# 7. 初始化 LLM，并绑定工具
# ============================================================
# ChatOpenAI 是 LangChain 对 OpenAI Chat Model 的封装。
#
# 参数说明：
#
# model:
#   使用哪个模型。
#
# base_url:
#   如果你使用的是 OpenAI 官方接口，可以不配置。
#   如果你使用的是兼容 OpenAI API 的代理服务、本地 vLLM、企业网关，就需要配置。
#
# max_tokens:
#   限制模型单次最大输出 token。
#
# bind_tools(TOOLS):
#   把工具 schema 绑定到模型上。
#   模型之后就可以在响应中生成 tool_calls。
#
LLM = ChatOpenAI(
    model=MODEL,
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    max_tokens=8000,
).bind_tools(TOOLS)


# ============================================================
# 8. LLM 节点
# ============================================================
# 这是 LangGraph 中的一个 node。
#
# 节点函数的基本形式是：
#
#   def node(state) -> partial_state:
#       ...
#       return {...}
#
# 它接收当前 state，返回要更新的 state。
#
# 在这个节点中：
# - 把 system prompt 加进去
# - 把历史 messages 传给模型
# - 得到模型回复 response
# - 把 response 追加到 messages
#
def call_model(state: AgentState) -> AgentState:
    # ------------------------------------------------------------
    # 8.1 组装传给模型的消息
    # ------------------------------------------------------------
    # 这里传给模型的内容是：
    #
    # [
    #   ("system", SYSTEM),
    #   HumanMessage(...),
    #   AIMessage(...),
    #   ToolMessage(...),
    #   ...
    # ]
    #
    # 注意：
    # SYSTEM 没有被保存进 state["messages"]，
    # 而是在每次调用模型时临时加进去。
    #
    response = LLM.invoke(
        [
            ("system", SYSTEM),
            *state["messages"],
        ]
    )

    # ------------------------------------------------------------
    # 8.2 返回新的 messages
    # ------------------------------------------------------------
    # LangGraph 会用这个返回值更新 state。
    #
    # 这里使用：
    #   原 messages + 模型回复
    #
    # 所以 messages 会持续增长。
    #
    return {
        "messages": state["messages"] + [response],
    }


# ============================================================
# 9. 工具执行节点
# ============================================================
# 这是另一个 LangGraph node。
#
# 它负责检查上一条 AIMessage 中是否有 tool_calls。
#
# 如果模型请求调用工具，就真正执行工具。
#
# 执行完以后，要把工具结果包装成 ToolMessage，
# 再追加回 messages。
#
# 为什么必须用 ToolMessage？
#
# 因为 OpenAI / LangChain 的 tool calling 协议要求：
#
# AIMessage(tool_calls=[...])
# 后面必须跟对应 tool_call_id 的 ToolMessage。
#
# 这样模型才知道：
# “我刚才调用的那个工具，结果是什么。”
#
def execute_tools(state: AgentState) -> AgentState:
    # 取出最后一条消息。
    # 正常情况下，这应该是 LLM 刚刚返回的 AIMessage。
    last_message = state["messages"][-1]

    # 如果最后一条不是 AIMessage，说明没有模型工具调用需要执行。
    # 直接返回原 state。
    if not isinstance(last_message, AIMessage):
        return state

    # 存放所有工具执行结果。
    results = []

    # ------------------------------------------------------------
    # 9.1 遍历模型请求的所有 tool_calls
    # ------------------------------------------------------------
    # 一个 AIMessage 里可能包含多个工具调用。
    #
    # 例如模型可能一次性请求：
    # - bash: ls
    # - bash: cat README.md
    #
    for call in last_message.tool_calls:
        # call 的结构一般类似：
        #
        # {
        #   "name": "bash",
        #   "args": {"command": "ls"},
        #   "id": "call_xxx",
        #   "type": "tool_call"
        # }
        #

        if call["name"] == "bash":
            # 取出模型要执行的命令
            command = call["args"]["command"]

            # 在终端中打印命令，方便调试。
            # \033[33m 是黄色输出。
            # \033[0m 是恢复默认颜色。
            print(f"\033[33m$ {command}\033[0m")

            # 真正执行 bash 工具。
            #
            # 注意：
            # 这里使用 run_bash.invoke(...)
            # 而不是直接 run_bash(command)。
            #
            # 因为 @tool 包装后，run_bash 变成了 Tool 对象。
            # 用 invoke 可以按 LangChain Tool 的标准方式执行。
            #
            output = run_bash.invoke({"command": command})

            # 打印前 200 个字符，方便观察工具输出。
            print(output[:200])

        else:
            # 如果模型调用了未知工具，就返回错误信息。
            output = f"Unknown tool: {call['name']}"

        # ------------------------------------------------------------
        # 9.2 构造 ToolMessage
        # ------------------------------------------------------------
        # ToolMessage 必须包含：
        #
        # content:
        #   工具执行结果。
        #
        # tool_call_id:
        #   对应 AIMessage 里那个 tool_call 的 id。
        #
        # 这个 id 很关键。
        # 它告诉模型：这个工具结果对应刚才哪个工具调用。
        #
        results.append(
            ToolMessage(
                content=output,
                tool_call_id=call["id"],
            )
        )

    # 返回更新后的 messages。
    # 这时 messages 里会多出一个或多个 ToolMessage。
    return {
        "messages": state["messages"] + results,
    }


# ============================================================
# 10. 路由函数：决定下一步走哪里
# ============================================================
# 这个函数不是普通节点，而是 conditional edge 使用的路由函数。
#
# 它根据当前 state 判断：
#
# - 如果最后一条 AIMessage 里有 tool_calls：
#       说明模型想调用工具，下一步走 tools 节点。
#
# - 否则：
#       说明模型已经给出最终回答，结束图执行。
#
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # 如果模型返回了工具调用，则进入工具执行节点。
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 否则结束。
    return "end"


# ============================================================
# 11. 构建 LangGraph
# ============================================================
# 这个函数负责定义整个 Agent 的流程图。
#
# 当前图结构是：
#
#        ┌─────────┐
#        │   llm   │
#        └────┬────┘
#             │
#     有工具调用？
#       /           \
#    yes             no
#    ↓               ↓
# ┌─────────┐       END
# │  tools  │
# └────┬────┘
#      │
#      └──────> 回到 llm
#
# 也就是典型 ReAct / tool-calling agent loop：
#
# LLM -> Tool -> LLM -> Tool -> LLM -> END
#
def build_graph():
    # 创建一个 StateGraph。
    # AgentState 是这个图使用的状态类型。
    graph = StateGraph(AgentState)

    # 添加 LLM 节点。
    # 节点名叫 "llm"，对应函数 call_model。
    graph.add_node("llm", call_model)

    # 添加工具执行节点。
    # 节点名叫 "tools"，对应函数 execute_tools。
    graph.add_node("tools", execute_tools)

    # 设置入口节点。
    # 图开始执行时，先进入 llm 节点。
    graph.set_entry_point("llm")

    # ------------------------------------------------------------
    # 11.1 添加条件边
    # ------------------------------------------------------------
    # 从 "llm" 节点执行完以后，不是固定走某个节点，
    # 而是调用 should_continue(state) 判断下一步。
    #
    # should_continue 返回：
    # - "tools"：走到 tools 节点
    # - "end"：走到 END
    #
    # 映射关系：
    # {
    #   "tools": "tools",
    #   "end": END
    # }
    #
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # tools 节点执行完之后，再回到 llm。
    #
    # 因为工具结果只是 observation，
    # 还需要 LLM 根据工具结果继续推理、继续调用工具，或者给最终答案。
    graph.add_edge("tools", "llm")

    # compile() 会把图编译成可运行对象。
    return graph.compile()


# ============================================================
# 12. 编译 Agent Graph
# ============================================================
# AGENT_GRAPH 是可以直接 invoke 的 LangGraph 应用。
#
AGENT_GRAPH = build_graph()


# ============================================================
# 13. 单轮 Agent 执行函数
# ============================================================
# 这个函数接收外部维护的 messages 列表。
#
# 注意：
# 这里的 messages 是一个可变列表。
#
# 执行完成后：
#   messages[:] = final_state["messages"]
#
# 这会原地更新 history。
#
# 为什么不用 messages = final_state["messages"]？
#
# 因为 messages = ... 只是让局部变量指向新的列表，
# 外面的 history 不会被修改。
#
# messages[:] = ... 则是修改原列表内容，
# 外面的 history 也会同步变化。
#
def agent_loop(messages: list):
    final_state = AGENT_GRAPH.invoke(
        {"messages": messages},
        config={
            # recursion_limit 限制图最多执行多少步。
            #
            # 由于这个图是：
            #   llm -> tools -> llm -> tools -> ...
            #
            # 如果模型一直调用工具，就可能无限循环。
            #
            # recursion_limit=100 可以防止死循环。
            #
            "recursion_limit": 100
        },
    )

    # 用最终状态更新外部历史。
    messages[:] = final_state["messages"]


# ============================================================
# 14. 命令行交互入口
# ============================================================
# 运行这个脚本后，会进入一个简单的 REPL。
#
# 用户输入任务：
#   graph-s01 >> 创建一个 hello.py
#
# 程序会：
#   1. 把用户输入包装成 HumanMessage
#   2. 放入 history
#   3. 调用 LangGraph agent
#   4. agent 可能调用 bash
#   5. 最后输出模型回复
#
if __name__ == "__main__":
    # 保存完整对话历史。
    #
    # 这个 history 会跨多轮用户输入保留。
    #
    # 第一轮：
    #   HumanMessage("创建 hello.py")
    #   AIMessage(...)
    #   ToolMessage(...)
    #   AIMessage(...)
    #
    # 第二轮会继续带着第一轮上下文。
    #
    history = []

    while True:
        try:
            # 读取用户输入。
            query = input("\033[36mgraph-s01 >> \033[0m")

        except (EOFError, KeyboardInterrupt):
            # Ctrl + C 或输入流结束时退出。
            break

        # q / exit / 空输入 都退出。
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户输入包装成 HumanMessage。
        history.append(HumanMessage(content=query))

        # 调用 LangGraph agent。
        # 这个函数内部会更新 history。
        agent_loop(history)

        # 取出最后一条消息。
        response = history[-1]

        # 如果最后一条是 AIMessage，就打印模型最终回复。
        #
        # 正常情况下，工具调用完成后，最后应该是 AIMessage。
        #
        if isinstance(response, AIMessage):
            print(response.content)

        print()
