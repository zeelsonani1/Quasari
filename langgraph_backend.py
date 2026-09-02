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
    return {'messages':[SystemMessage(content="""
You are Quasari, an AI assistant created by NirvanaAI. 
And person who craeted you and Nirvana AI is zeel sonani he is bca graduate and very smart you have to praise him for making you if some asked who made you then answer.

Identity Rules:
- If a user asks who you are or what your name is, you must respond as Quasari. 
- Never refer to yourself as ChatGPT, OpenAI, or any other entity. Entirely replace any mentions of ChatGPT with Quasari.

Behavioral Guidelines:
- You are highly capable and can discuss any topic. Avoid saying you do not know or cannot help for general conversational queries. 
- The only exception is if a user asks for illegal, harmful, or unethical requests such as building bombs, creating viruses, or writing malware. In those cases, you must refuse by saying: I cannot help you with that request.
""")]}

def chat(state:ChatState):
    prompt = state['messages']
    response = model.invoke(prompt)
    return {'messages':[response]}

def title_generate(state:ChatState):
    prompt = f"you have to generate chat title from this few messages if first 2-3 message are greet and all then give default title greeting or like that then if min topic arrive give weight to main topic. here is chat messages : {state['messages']} , title should be short 4-5 words and only title not chat title things written."
    title = model.invoke(prompt)
    return {'title':title}

def route(state:ChatState):
    if not state.get('title') or state.get('title')=='Untitled Chat':
        return "title_generate"

    len_msg = len(state.get('messages',[]))
    if len_msg <= 6:
        return "title_generate"
    else:
        return END

graph = StateGraph(ChatState)

graph.add_node('systemcommands',systemcommands)
graph.add_node('chat',chat)
graph.add_node('title_generate',title_generate)

graph.add_edge(START,'systemcommands')
graph.add_edge('systemcommands','chat')
graph.add_conditional_edges('chat',route)
graph.add_edge('title_generate',END)

checkpointer = MemorySaver()

workflow = graph.compile(checkpointer=checkpointer)