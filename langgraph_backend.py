from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
def chat_node(state: ChatState):
    ## take user query from state
    messages = state['messages']
    ## send to llm
    response = llm.invoke(messages)
    ## store the result
    return {'messages': [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)
