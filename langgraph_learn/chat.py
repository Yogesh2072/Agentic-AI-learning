from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

#add_messages is a built-in LangGraph reducer function.
#A 'reducer' determines how the existing state value and a new update should be combined
#Example:
#new_state = reducer(
#     existing_state,
#     new_update
# )
#
#Note: If reducer is not provided, the new update will replace the existing state value.
# What problem is add_messages solving?
# In a chatbot application, you often want to maintain a conversation history.
# Each time a new message is sent or received, you want to append it to the existing list of messages rather than replacing the entire list. The add_messages reducer function is designed to handle this
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return { "messages": [response] }

def samplenode(state: State):
    print("\n\nInside samplenode node", state)
    return { "messages": ["Sample Message Appended"] }

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)

graph = graph_builder.compile()

updated_state = graph.invoke(State({"messages": ["What is my name?"]}))
print("\n\nupdated_state", updated_state)

# (START) -> chatbot -> samplenode -> (END)

# state = { messages: ["Hey there"] }
# node runs: chatbot(state: ["Hey There"]) -> ["Hi, This is a message from ChatBot Node"]
# state = { "messages": ["Hey there", "Hi, This is a message from ChatBot Node"]  }