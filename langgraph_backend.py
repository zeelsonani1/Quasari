# langgraph_backend.py

import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
)
from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# MODEL
# ============================================================

MAX_OUTPUT_TOKENS = 2000

model = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=MAX_OUTPUT_TOKENS,
)

model_with_tools = None


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Quasari, a helpful AI assistant.

Your job is to:
- Give accurate and useful answers.
- Explain concepts clearly.
- Adapt explanations to the user's level.
- Use concise answers when the question is simple.
- Give detailed explanations when necessary.
- Use tools when current information is required.
- Never pretend to know something you do not know.
"""


# ============================================================
# STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str


# ============================================================
# SEARCH TOOL
# ============================================================

search_engine = DuckDuckGoSearchRun()


@tool("duckduckgo_search")
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    """
    return search_engine.run(query)


tools = [duckduckgo_search]

model_with_tools = model.bind_tools(tools)


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation.

    This is intentionally an approximation.
    Roughly 4 characters ~= 1 token for normal English text.
    """

    if not text:
        return 0

    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[BaseMessage]) -> int:
    total = 0

    for message in messages:
        if message.content:
            total += estimate_tokens(str(message.content))

    return total


# ============================================================
# CONVERSATION SUMMARIZATION
# ============================================================

def summarize_history(messages: list[BaseMessage]) -> str:

    history_text = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in messages
        if message.content
    )

    summary_prompt = f"""
Summarize the following conversation.

Keep:
- Important user information
- Important decisions
- Previous questions
- Important technical details
- Context needed to answer future questions

Remove:
- Repetition
- Unnecessary wording
- Small talk

Conversation:

{history_text}
"""

    summary = model.invoke(
        [HumanMessage(content=summary_prompt)]
    )

    return str(summary.content)


# ============================================================
# CHAT NODE
# ============================================================

def chat(state: ChatState):

    history = state["messages"]

    # --------------------------------------------------------
    # Context management
    # --------------------------------------------------------

    # Keep conversation reasonably small.
    #
    # IMPORTANT:
    # This is an application-level safety limit.
    # It prevents us from sending an enormous conversation
    # to the API.
    #
    if len(history) > 12:

        old_messages = history[:-8]
        recent_messages = history[-8:]

        try:

            summary = summarize_history(old_messages)

            payload = [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(
                    content=(
                        "Summary of the earlier conversation:\n"
                        + summary
                    )
                ),
                *recent_messages,
            ]

        except Exception:

            # If summarization itself fails,
            # fall back to recent messages.

            payload = [
                SystemMessage(content=SYSTEM_PROMPT),
                *recent_messages,
            ]

    else:

        payload = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history,
        ]

    # --------------------------------------------------------
    # Token safety check
    # --------------------------------------------------------

    input_tokens = estimate_message_tokens(payload)

    # Leave room for the model's answer.
    #
    # This number is intentionally conservative.
    MAX_INPUT_TOKENS = 10000

    if input_tokens > MAX_INPUT_TOKENS:

        # Try using only the most recent messages.

        recent_messages = history[-4:]

        payload = [
            SystemMessage(content=SYSTEM_PROMPT),
            *recent_messages,
        ]

        input_tokens = estimate_message_tokens(payload)

        if input_tokens > MAX_INPUT_TOKENS:

            raise RuntimeError(
                "CONTEXT_LIMIT_REACHED"
            )

    # --------------------------------------------------------
    # Model call
    # --------------------------------------------------------

    try:

        response = model_with_tools.invoke(payload)

        return {
            "messages": [response]
        }

    except Exception as e:

        error_text = str(e).lower()

        # ----------------------------------------------------
        # Rate limit / API limit
        # ----------------------------------------------------

        if (
            "rate limit" in error_text
            or "429" in error_text
            or "too many requests" in error_text
            or "quota" in error_text
        ):

            raise RuntimeError(
                "API_LIMIT_REACHED"
            ) from e

        # ----------------------------------------------------
        # Context/token limit
        # ----------------------------------------------------

        if (
            "token" in error_text
            or "context" in error_text
            or "maximum" in error_text
        ):

            raise RuntimeError(
                "CONTEXT_LIMIT_REACHED"
            ) from e

        # ----------------------------------------------------
        # Unknown error
        # ----------------------------------------------------

        raise RuntimeError(
            "MODEL_ERROR"
        ) from e


# ============================================================
# TITLE GENERATION
# ============================================================

def title_generate(state: ChatState):

    messages = state["messages"]

    clean_messages = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in messages
        if message.content
    )

    prompt = f"""
Create a short title for this conversation.

Rules:
- Maximum 6 words
- No quotation marks
- No punctuation at the end
- Describe the main topic
- Do not say "Chat" or "Conversation"

Conversation:

{clean_messages}
"""

    try:

        result = model.invoke(
            [HumanMessage(content=prompt)]
        )

        title = str(result.content).strip()

        if not title:
            title = "New Chat"

        return {
            "title": title
        }

    except Exception:

        return {
            "title": "New Chat"
        }


# ============================================================
# ROUTER
# ============================================================

def route(state: ChatState):

    last_message = state["messages"][-1]

    # --------------------------------------------------------
    # Tool call
    # --------------------------------------------------------

    if getattr(last_message, "tool_calls", None):

        return "tools"

    # --------------------------------------------------------
    # Generate title only once
    # --------------------------------------------------------

    title = state.get("title")

    if not title or title == "Untitled Chat":

        return "title_generate"

    # --------------------------------------------------------
    # Normal conversation finished
    # --------------------------------------------------------

    return END


# ============================================================
# GRAPH
# ============================================================

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


# START → CHAT

graph.add_edge(
    START,
    "chat"
)


# CHAT → TOOL / TITLE / END

graph.add_conditional_edges(
    "chat",
    route,
    {
        "tools": "tools",
        "title_generate": "title_generate",
        END: END,
    }
)


# TOOL → CHAT

graph.add_edge(
    "tools",
    "chat"
)


# TITLE → END

graph.add_edge(
    "title_generate",
    END
)


# ============================================================
# SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


workflow = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# LOAD USER THREADS
# ============================================================

def load_threads(user_id: str):

    all_threads = set()

    for check in checkpointer.list(None):

        config = check.config.get(
            "configurable",
            {}
        )

        full_thread_id = config.get(
            "thread_id",
            ""
        )

        if full_thread_id.startswith(
            f"{user_id}:"
        ):

            raw_thread_id = full_thread_id.split(
                ":",
                1
            )[1]

            all_threads.add(
                raw_thread_id
            )

    return list(all_threads)