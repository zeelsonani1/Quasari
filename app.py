import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import workflow
import uuid

# =========================== Utility functions ================

def generate_thread():
    return str(uuid.uuid4())

def new_chat():
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

# =========================== SessionState manage ==============


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()

if 'chat_thread_ids' not in st.session_state:
    st.session_state['chat_thread_ids'] = []

add_thread(st.session_state['thread_id'])

# =========================== sidebar ==========================

st.sidebar.title('Quasari')

if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_thread_ids'][::-1]:
    if st.sidebar.button(thread_id):
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
            for message_chunk, metadata in workflow.stream(input_data, config=CONFIG, stream_mode='messages'):

                if hasattr(message_chunk, 'type') and message_chunk.type == 'system':
                    continue
                
                from langchain_core.messages import SystemMessage
                if isinstance(message_chunk, SystemMessage):
                    continue
                    
                # Only yield actual text chunks meant for the user
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    yield message_chunk.content
        ai = st.write_stream(chunk_generator())

    st.session_state['message_history'].append({'role':'ai','content':ai})