import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph_backend import workflow, load_threads
import uuid
from streamlit_local_storage import LocalStorage


local_storage = LocalStorage()


# =========================== Page Config ======================

st.set_page_config(
    page_title="Quasari",
    page_icon="logo.png", 
    layout="wide"
)

# =========================== Utility functions ================

def generate_user():
    uid = f"guest_{uuid.uuid4().hex[:8]}"
    return uid

def add_user(user):
    if user not in st.session_state['user_ids']:
        st.session_state['user_ids'].append(user)

def generate_thread():
    return str(uuid.uuid4())

def new_chat():
    if st.session_state.get('message_history', []) != []:
        thread_id = generate_thread()
        st.session_state['thread_id'] = thread_id
        add_thread(thread_id)
        st.session_state['message_history'] = []
        st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread_ids']:
        st.session_state['chat_thread_ids'].append(thread_id)

def load_conversations(compound_id):
    state = workflow.get_state(config={'configurable': {'thread_id': compound_id}})
    if not state.values or 'messages' not in state.values:
        return []
    return state.values['messages']

def load_title(compound_id):
    state = workflow.get_state(config={'configurable': {'thread_id': compound_id}})
    if not state.values or 'title' not in state.values:
        return "Untitled Chat"
    
    # Handle both string titles and message-object types securely
    title_obj = state.values['title']
    return title_obj.content if hasattr(title_obj, 'content') else str(title_obj)

# =========================== SessionState manage ==============

if 'user_ids' not in st.session_state:
    st.session_state['user_ids'] = []

if 'user' not in st.session_state:
    params = st.query_params
    stored_uid = local_storage.getItem("quasari_uid")
    if 'uid' in params:
        st.session_state['user'] = params['uid']
        local_storage.setItem("quasari_uid", params['uid'])
    elif stored_uid:
        st.session_state['user'] = stored_uid
        st.query_params['uid'] = stored_uid
    else:
        new_uid = generate_user()
        st.session_state['user'] = new_uid
        st.query_params['uid'] = new_uid
        local_storage.setItem("quasari_uid", new_uid)

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()

if 'chat_thread_ids' not in st.session_state:
    st.session_state['chat_thread_ids'] = load_threads(st.session_state['user'])

add_thread(st.session_state['thread_id'])
add_user(st.session_state['user'])


# =========================== sidebar ==========================

st.markdown("""
<style>
.quasari-title {
    font-size: 30px;
    font-weight: 800;
    padding: 5px 0 20px 0;
    background: linear-gradient(90deg, #8B5CF6, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 18px;
    background: linear-gradient(90deg, #7C3AED, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* New Chat button */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border: 1px solid rgba(124, 58, 237, 0.45);
    border-radius: 12px;
    padding: 10px 16px;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(34, 211, 238, 0.12));
    color: white;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.25s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #22D3EE;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.35), rgba(34, 211, 238, 0.25));
    transform: translateY(-1px);
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: scale(0.98);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Quasari")

if st.sidebar.button("＋ New Chat"):
    new_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_thread_ids'][::-1]:
    compound_key = f"{st.session_state['user']}:{thread_id}"
    
    title = load_title(compound_id=compound_key)
    
    if st.sidebar.button(title, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        message = load_conversations(compound_id=compound_key)

        temp_messages = []
        for msg in message:
            if isinstance(msg, SystemMessage):
                continue
            role = 'user' if isinstance(msg, HumanMessage) else 'ai'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()

# =========================== message history load =============

for message in st.session_state['message_history']:
    if "You are Quasari" in message['content']:
        continue
    if message['role'] == 'system':
        continue
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# =========================== conversation =====================

CONFIG = {'configurable': {'thread_id': f"{st.session_state['user']}:{st.session_state['thread_id']}"},
          'metadata':{'thread_id':st.session_state['thread_id']},
          'run_name':['quasari']}

user = st.chat_input('Type Here.')

if user:
    st.session_state['message_history'].append({'role': 'user', 'content': user})
    with st.chat_message('user'):
        st.markdown(user)

    with st.chat_message('ai'):
        input_data = {'messages': [HumanMessage(content=user)]}
        
        def chunk_generator():
            for message_chunk, metadata in workflow.stream(input_data, config=CONFIG, stream_mode='messages'):
                if metadata.get('langgraph_node') != 'chat':
                    continue
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    yield message_chunk.content

        ai = st.write_stream(chunk_generator())
        
    if ai:
        st.session_state['message_history'].append({'role': 'ai', 'content': ai})
        st.rerun()