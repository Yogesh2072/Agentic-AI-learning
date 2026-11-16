from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI(
    api_key="AIzaSyBe957AmpWDzXIx7X1mhBmrFl3dp3bop1w",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thought.
    You work on START, PLAN and OUPUT steps.
    You need to first PLAN what needs to be done . The PLAN can be multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.

    Rules:
    - Strictly Follow the given JSON output format
    - Only run one step at a time.
    - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to the displayed to the user).

    Output JSON Format:
    { "step": "START" | "PLAN" | "OUTPUT", "content": "string" }

    Example:
    START: Hey, Can you solve 2 + 3 * 5 / 10
    PLAN: { "step": "PLAN": "content": "Seems like user is interested in math problem" }
    PLAN: { "step": "PLAN": "content": "looking at the problem, we should solve this using BODMAS method" }
    PLAN: { "step": "PLAN": "content": "Yes, The BODMAS is correct thing to be done here" }
    PLAN: { "step": "PLAN": "content": "first we must multiply 3 * 5 which is 15" }
    PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 15 / 10" }
    PLAN: { "step": "PLAN": "content": "We must perform divide that is 15 / 10  = 1.5" }
    PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 1.5" }
    PLAN: { "step": "PLAN": "content": "Now finally lets perform the add 3.5" }
    PLAN: { "step": "PLAN": "content": "Great, we have solved and finally left with 3.5 as ans" }
    OUTPUT: { "step": "OUTPUT": "content": "3.5" }
    
"""

print("\n\n\n")

message_history = [
    { "role": "system", "content": SYSTEM_PROMPT },
]

user_query = input("👉🏻 ")
message_history.append({ "role": "user", "content": user_query })

while True:
    response = client.chat.completions.create(
        model="gemini-3.8-flash",
        response_format={"type": "json_object"},
        messages=message_history
    )

    raw_result = response.choices[0].message.content
    # message_history.append({"role": "assistant", "content": raw_result})
    
    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START":
        print("🔥", parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "PLAN":
        message_history.append({
            "role": "assistant",
            "content": raw_result
        })
#         That is correct for recording Gemini's response. However, your next API request sends this history back to Gemini, and the last message is an assistant message.
# Gemini's API can reject this conversation sequence because it expects a user message to initiate the next turn.
        message_history.append({
            "role": "user",
            "content": "Provide the next step in the required JSON format."
        })
        print("🧠", parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "OUTPUT":
        print("🤖", parsed_result.get("content"))
        break

print("\n\n\n")
# response = client.chat.completions.create(
#     model="gemini-3.8-flash",
#     n=1,
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": "Help me write a code to find the factorial of a number in Python"
#         },
#         {"role": "user", "content": json.dumps({"step": "PLAN", "content": "The user is asking for a clear and accessible explanation of how Artificial Intelligence (AI) works"})},
#         {"role": "user", "content": json.dumps({"step": "PLAN", "content": "To find the factorial of a number in Python, we can implement an iterative function using a loop and also demonstrate Python's built-in math.factorial function." })}
#     ]
# )

# print(response.choices[0].message.content)
