import streamlit as st
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph_backend import workflow,load_threads
import uuid

# =========================== Page Config ======================

st.set_page_config(
    page_title="Quasari",
    page_icon="logo.png", 
    layout="wide"
)

# =========================== Utility functions ================

def generate_thread():
    return str(uuid.uuid4())

def new_chat():
    if st.session_state['message_history'] != []:
        thread_id = generate_thread()
        st.session_state['thread_id']=thread_id
        add_thread(st.session_state['thread_id'])
        st.session_state['message_history']=[]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread_ids']:
        st.session_state['chat_thread_ids'].append(thread_id)

def load_conversations(thread_id):
    state = workflow.get_state(config={'configurable':{'thread_id':thread_id}})
    if not state.values or 'messages' not in state.values:
        return []
    return state.values['messages']

def load_title(thread_id):
    state = workflow.get_state(config={'configurable':{'thread_id':thread_id}})
    if not state.values or 'title' not in state.values:
        return "Unititled Chat"
    return state.values['title'].content

# =========================== SessionState manage ==============

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = f"guest_{uuid.uuid4().hex[:8]}"

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()

if 'chat_thread_ids' not in st.session_state:
    st.session_state['chat_thread_ids'] = load_threads(st.session_state['user_id'])

add_thread(st.session_state['thread_id'])

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

    background: linear-gradient(
        135deg,
        rgba(124, 58, 237, 0.18),
        rgba(34, 211, 238, 0.12)
    );

    color: white;
    font-weight: 600;
    font-size: 15px;

    transition: all 0.25s ease;
}

/* Hover */
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #22D3EE;

    background: linear-gradient(
        135deg,
        rgba(124, 58, 237, 0.35),
        rgba(34, 211, 238, 0.25)
    );

    transform: translateY(-1px);
}

/* Click */
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
    title = load_title(thread_id=thread_id)
    if st.sidebar.button(title):
        st.session_state['thread_id']=thread_id
        message = load_conversations(thread_id=thread_id)

        temp_messages = []

        for msg in message:
            if isinstance(msg,HumanMessage):
                role = 'user'
            else:
                role = 'ai'
            temp_messages.append({'role':role,'content':msg.content})

        st.session_state['message_history']=temp_messages


# =========================== message history load =============

for message in st.session_state['message_history']:
    if "You are Quasari" in message['content']:
        continue
    if message['role'] == 'system':
        continue
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# =========================== conversation =====================


CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}


user = st.chat_input('Type Here.')

if user:
    st.session_state['message_history'].append({'role':'user','content':user})
    with st.chat_message('user'):
        st.text(user)

    with st.chat_message('ai'):
        input_data = {'messages': [HumanMessage(content=user)]}
        
        def chunk_generator():
            seen_message_ids = []
            current_message_chunks = {}
            
            for message_chunk, metadata in workflow.stream(input_data, config=CONFIG, stream_mode='messages'):
                if metadata.get('langgraph_node') != 'chat':
                    continue
                    
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    msg_id = getattr(message_chunk, 'id', None) or metadata.get('message_id')
                    
                    if msg_id:
                        if msg_id not in seen_message_ids:
                            seen_message_ids.append(msg_id)
                            current_message_chunks[msg_id] = []
                        
                        current_message_chunks[msg_id].append(message_chunk.content)
                        
                        if msg_id == seen_message_ids[-1] and len(seen_message_ids) > 1:
                            yield message_chunk.content
                            
            if len(seen_message_ids) == 1:
                for chunk in current_message_chunks[seen_message_ids[0]]:
                    yield chunk
        ai = st.write_stream(chunk_generator())
    if ai:
        st.session_state['message_history'].append({'role':'ai','content':ai})