from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def systemcommands(state:ChatState):
    return {'messages':[SystemMessage(content='you are Quasari if user ask who are you and what is your name so you have to response Quasari not ChatGPT or else , basically replace chatgpt with Quasari. Created by NirvanaAI.')]}

def chat(state:ChatState):
    prompt = state['messages']
    response = model.invoke(prompt)
    return {'messages':[response]}

graph = StateGraph(ChatState)

graph.add_node('systemcommands',systemcommands)
graph.add_node('chat',chat)

graph.add_edge(START,'systemcommands')
graph.add_edge('systemcommands','chat')
graph.add_edge('chat',END)

checkpointer = MemorySaver()

workflow = graph.compile(checkpointer=checkpointer)