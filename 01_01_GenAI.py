from google import genai

client = genai.Client(api_key="AIzaSyBe957AmpWDzXIx7X1mhBmrFl3dp3bop1w")

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain how AI works in a few words, who are you ?"
)
print(interaction.output_text)