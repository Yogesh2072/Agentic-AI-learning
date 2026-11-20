from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

load_dotenv()

client = OpenAI()

#the model used in this code is gemini-2.5-flash which is a Google Gemini API model.
#  You can also use other models like gemini-3.7-flash, gemini-3.8-flash, etc. 
evaluator = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]

def chatbot(state: State):
    print("ChatBot Node", state)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            { "role": "user", "content": state.get("user_query") }
        ]
    )

    state["llm_output"] = response.choices[0].message.content
    return state
# evalaute_response is a function that evaluates the response from the chatbot node and determines the next node to transition to in the state graph. 
# It takes the current state as input and returns a string indicating the next node to transition to. In this case, it checks if the response from the 
# chatbot is good or not (currently hardcoded to always return "chatbot_gemini"). If it were to return "endnode", it would indicate that the conversation should end.
def evalaute_response(state: State) -> Literal["chatbot_gemini", "endnode"]:
    print("evalaute_response Node", state)
    # The following prompt is constructed to evaluate the response from the chatbot node. 
    # It includes the user's question and the LLM's response, and asks the evaluator to assess whether the response is good or bad 
    # based on specific criteria. The evaluator is expected to respond with either "GOOD" or "BAD", which will determine the next node in the state graph.
    prompt = f"""
    You are an evaluator.

    User Question:
    {state['user_query']}

    LLM Response:
    {state['llm_output']}

    Evaluate whether the response:
    1. Answers the user's question.
    2. Is factually correct.
    3. Is relevant.
    4. Is sufficiently clear.

    Respond with only GOOD or BAD.
    """

    result = evaluator.invoke(prompt)

    decision = result.content[0]["text"].strip().upper()

    if decision == "GOOD":
        return "endnode"

    return "chatbot_gemini"

def chatbot_gemini(state: State):
    print("chatbot_gemini Node", state)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            { "role": "user", "content": state.get("user_query") }
        ]
    )

    state["llm_output"] = response.choices[0].message.content
    return state

def endnode(state: State):
    print("endnode Node", state)
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chatbot_gemini", chatbot_gemini)
graph_builder.add_node("endnode", endnode)


graph_builder.add_edge(START, "chatbot")
# DEBUG: The following line is added to create conditional edges based on the evaluation of the response from the chatbot node.
graph_builder.add_conditional_edges("chatbot", evalaute_response)

graph_builder.add_edge("chatbot_gemini", "endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()

# Demonstration of how to invoke the graph with an initial state
updated_state = graph.invoke(State({"user_query": "Hey, Who is John Doe?"}))
print(updated_state)