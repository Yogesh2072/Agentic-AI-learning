from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key="AIzaSyBe957AmpWDzXIx7X1mhBmrFl3dp3bop1w",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

SYSTEM_PROMPT = """You are a helpful assistant that translates natural language into code."""

response = client.chat.completions.create(
    model="gemini-3.8-flash",
    n=1,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
    ]
)

print(response.choices[0].message.content)
