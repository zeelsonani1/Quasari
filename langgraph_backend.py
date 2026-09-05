import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


# =========================================================
# MODEL
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Quasari, an AI assistant created by NirvanaAI.

Your creator is Zeel Sonani, a BCA graduate.

Identity:
- If asked who you are, say you are Quasari.
- Never identify yourself as ChatGPT or OpenAI.

Behavior:
- Be helpful, friendly and conversational.
- Give clear and useful answers.
- For illegal, harmful, or unethical requests, refuse briefly.
"""


# =========================================================
# STATE
# =========================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    title: str


# =========================================================
# TOOLS
# =========================================================

search_engine = DuckDuckGoSearchRun()


@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web for current or up-to-date information.
    """
    return search_engine.run(query)


tools = [
    duckduckgo_search
]

model_with_tools = model.bind_tools(tools)


# =========================================================
# CHAT
# =========================================================

def chat(state: ChatState):

    messages = state["messages"]

    # -----------------------------------------------------
    # Keep context manageable
    # -----------------------------------------------------

    if len(messages) > 12:

        old_messages = messages[:-8]
        recent_messages = messages[-8:]

        history_text = "\n".join(
            f"{type(message).__name__}: {message.content}"
            for message in old_messages
            if message.content
        )

        summary_prompt = f"""
Summarize the previous conversation.

Keep:
- important facts
- user preferences
- decisions
- important context
- important questions

Be concise.

Previous conversation:

{history_text}
"""

        summary = model.invoke([
            HumanMessage(content=summary_prompt)
        ]).content

        payload = [
            SystemMessage(
                content=f"""
{SYSTEM_PROMPT}

Previous conversation summary:
{summary}
"""
            )
        ]

        payload.extend(recent_messages)

    else:

        payload = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ]

        payload.extend(messages)


    response = model_with_tools.invoke(payload)

    return {
        "messages": [response]
    }


# =========================================================
# TITLE GENERATOR
# =========================================================

def title_generate(state: ChatState):

    messages = state["messages"]

    conversation = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in messages
        if not isinstance(message, SystemMessage)
        and message.content
    )

    prompt = f"""
Create a short title for this conversation.

Rules:
- 3 to 5 words
- Only output the title
- No quotation marks
- Don't write "Chat Title:"
- If the conversation is only greetings, use "Greeting"
- Focus on the main topic

Conversation:

{conversation}
"""

    title = model.invoke([
        HumanMessage(content=prompt)
    ]).content.strip()

    return {
        "title": title
    }


# =========================================================
# ROUTER
# =========================================================

def route(state: ChatState):

    messages = state["messages"]

    last_message = messages[-1]

    # -----------------------------------------------------
    # Tool call
    # -----------------------------------------------------

    if getattr(last_message, "tool_calls", None):

        return "tools"


    # -----------------------------------------------------
    # Generate title once
    # -----------------------------------------------------

    if not state.get("title"):

        human_messages = [
            message
            for message in messages
            if isinstance(message, HumanMessage)
        ]

        if human_messages:

            return "title_generate"


    return END


# =========================================================
# GRAPH
# =========================================================

graph = StateGraph(ChatState)


graph.add_node(
    "chat",
    chat
)

graph.add_node(
    "tools",
    ToolNode(tools)
)

graph.add_node(
    "title_generate",
    title_generate
)


graph.add_edge(
    START,
    "chat"
)


# ONE router only
graph.add_conditional_edges(
    "chat",
    route,
    {
        "tools": "tools",
        "title_generate": "title_generate",
        END: END,
    }
)


graph.add_edge(
    "tools",
    "chat"
)


graph.add_edge(
    "title_generate",
    END
)


# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


workflow = graph.compile(
    checkpointer=checkpointer
)


# =========================================================
# LOAD USER THREADS
# =========================================================

def load_threads(user_id):

    threads = set()

    for checkpoint in checkpointer.list(None):

        config = checkpoint.config

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id", "")
        )

        if thread_id.startswith(f"{user_id}:"):

            raw_thread_id = thread_id.split(
                ":",
                1
            )[1]

            threads.add(raw_thread_id)

    return list(threads)