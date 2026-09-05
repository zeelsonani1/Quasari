import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from streamlit_local_storage import LocalStorage

from langgraph_backend import (
    workflow,
    load_threads,
)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Quasari",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# LOCAL STORAGE
# =========================================================

local_storage = LocalStorage()


# =========================================================
# HELPERS
# =========================================================

def generate_user_id():

    return f"guest_{uuid.uuid4().hex}"


def generate_thread_id():

    return str(uuid.uuid4())


def get_compound_thread_id(thread_id):

    return (
        f"{st.session_state.user_id}:"
        f"{thread_id}"
    )


def get_config(thread_id=None):

    if thread_id is None:
        thread_id = st.session_state.thread_id

    return {
        "configurable": {
            "thread_id": get_compound_thread_id(
                thread_id
            )
        },
        "metadata": {
            "user_id": st.session_state.user_id,
            "thread_id": thread_id,
        },
    }


# =========================================================
# USER ID
# =========================================================

def initialize_user():

    # Already initialized in this Streamlit session
    if "user_id" in st.session_state:
        return


    # Try browser localStorage
    stored_user = local_storage.getItem(
        "quasari_user_id"
    )


    if stored_user:

        st.session_state.user_id = stored_user

    else:

        new_user = generate_user_id()

        st.session_state.user_id = new_user

        local_storage.setItem(
            "quasari_user_id",
            new_user
        )


# =========================================================
# THREAD
# =========================================================

def initialize_thread():

    if "thread_id" not in st.session_state:

        st.session_state.thread_id = (
            generate_thread_id()
        )


def initialize_history():

    if "message_history" not in st.session_state:

        st.session_state.message_history = []


def initialize_threads():

    if "chat_thread_ids" not in st.session_state:

        st.session_state.chat_thread_ids = (
            load_threads(
                st.session_state.user_id
            )
        )


# =========================================================
# LOAD CHAT
# =========================================================

def load_conversation(thread_id):

    state = workflow.get_state(
        get_config(thread_id)
    )

    if not state.values:

        return []

    return state.values.get(
        "messages",
        []
    )


def load_title(thread_id):

    state = workflow.get_state(
        get_config(thread_id)
    )

    if not state.values:

        return "New Chat"

    title = state.values.get("title")

    if not title:

        return "New Chat"

    return str(title)


# =========================================================
# UI MESSAGES
# =========================================================

def convert_messages(messages):

    result = []

    for message in messages:

        if isinstance(
            message,
            SystemMessage
        ):
            continue

        if isinstance(
            message,
            HumanMessage
        ):
            role = "user"
        else:
            role = "assistant"


        content = message.content

        if isinstance(content, list):

            content = "\n".join(
                str(item)
                for item in content
            )


        result.append({
            "role": role,
            "content": content,
        })

    return result


# =========================================================
# NEW CHAT
# =========================================================

def new_chat():

    st.session_state.thread_id = (
        generate_thread_id()
    )

    st.session_state.message_history = []

    st.rerun()


# =========================================================
# OPEN CHAT
# =========================================================

def open_chat(thread_id):

    st.session_state.thread_id = thread_id

    messages = load_conversation(
        thread_id
    )

    st.session_state.message_history = (
        convert_messages(messages)
    )

    st.rerun()


# =========================================================
# INITIALIZATION
# =========================================================

initialize_user()
initialize_thread()
initialize_history()
initialize_threads()


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 1100px;
}


/* Sidebar */

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.15);
}


[data-testid="stSidebar"] h1 {

    font-size: 28px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #8B5CF6,
        #22D3EE
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


[data-testid="stSidebar"] .stButton button {

    width: 100%;
    border-radius: 10px;

    border: 1px solid rgba(
        124,
        58,
        237,
        0.25
    );

    transition: 0.2s ease;
}


[data-testid="stSidebar"] .stButton button:hover {

    border-color: #22D3EE;
    transform: translateY(-1px);
}


/* Chat input */

[data-testid="stChatInput"] {

    border-radius: 16px;
}


/* Chat messages */

[data-testid="stChatMessage"] {

    border-radius: 14px;
}


/* Empty screen */

.quasari-home {

    text-align: center;
    padding-top: 15vh;
}


.quasari-logo {

    font-size: 58px;
    font-weight: 900;

    background: linear-gradient(
        90deg,
        #8B5CF6,
        #22D3EE
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.quasari-subtitle {

    font-size: 18px;
    opacity: 0.65;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Quasari")


if st.sidebar.button(
    "＋  New Chat",
    use_container_width=True,
):

    new_chat()


st.sidebar.divider()

st.sidebar.subheader(
    "Conversations"
)


for thread_id in reversed(
    st.session_state.chat_thread_ids
):

    title = load_title(thread_id)

    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}",
        use_container_width=True,
    ):

        open_chat(thread_id)


# =========================================================
# HOME SCREEN
# =========================================================

if not st.session_state.message_history:

    st.markdown(
        """
<div class="quasari-home">

<div class="quasari-logo">
Quasari
</div>

<div class="quasari-subtitle">
Your AI assistant
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# HISTORY
# =========================================================

for message in (
    st.session_state.message_history
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Message Quasari..."
)


if prompt:

    # -----------------------------------------------------
    # USER
    # -----------------------------------------------------

    st.session_state.message_history.append({
        "role": "user",
        "content": prompt,
    })


    with st.chat_message("user"):

        st.markdown(prompt)


    # -----------------------------------------------------
    # AI
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        def response_generator():

            input_data = {
                "messages": [
                    HumanMessage(
                        content=prompt
                    )
                ]
            }


            for chunk, metadata in workflow.stream(
                input_data,
                config=get_config(),
                stream_mode="messages",
            ):

                if (
                    metadata.get(
                        "langgraph_node"
                    )
                    != "chat"
                ):
                    continue


                content = getattr(
                    chunk,
                    "content",
                    None
                )


                if content:

                    yield content


        response = st.write_stream(
            response_generator()
        )


    # -----------------------------------------------------
    # SAVE UI HISTORY
    # -----------------------------------------------------

    if response:

        st.session_state.message_history.append({
            "role": "assistant",
            "content": response,
        })


        # Add thread after first successful message
        if (
            st.session_state.thread_id
            not in st.session_state.chat_thread_ids
        ):

            st.session_state.chat_thread_ids.append(
                st.session_state.thread_id
            )


        st.rerun()