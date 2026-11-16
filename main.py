import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
text = "Hello, world!"
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")

decoded_text = encoding.decode([9906, 11, 1917, 0])
print(f"Decoded Text: {decoded_text}")