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
    title : str

def systemcommands(state:ChatState):
    return {'messages':[SystemMessage(content='you are Quasari if user ask who are you and what is your name so you have to response Quasari not ChatGPT or else , basically replace chatgpt with Quasari. Created by NirvanaAI.')]}

def chat(state:ChatState):
    prompt = state['messages']
    response = model.invoke(prompt)
    return {'messages':[response]}

def title_generate(state:ChatState):
    prompt = f"you have to generate chat title from this few messages if first 2-3 message are greet and all then give default title greeting or like that then if min topic arrive give weight to main topic. here is chat messages : {state['messages']} , title should be short 4-5 words and only title not chat title things written."
    title = model.invoke(prompt)
    return {'title':title}

graph = StateGraph(ChatState)

graph.add_node('systemcommands',systemcommands)
graph.add_node('chat',chat)
graph.add_node('title_generate',title_generate)

graph.add_edge(START,'systemcommands')
graph.add_edge('systemcommands','chat')
graph.add_edge('chat','title_generate')
graph.add_edge('title_generate',END)

checkpointer = MemorySaver()

workflow = graph.compile(checkpointer=checkpointer)