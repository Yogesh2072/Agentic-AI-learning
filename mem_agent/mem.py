from dotenv import load_dotenv
from mem0 import Memory
import os
import json

from openai import OpenAI

load_dotenv()

client = OpenAI()
# here we are fetching the OpenAI API key from the environment variables.
# This key is essential for authenticating requests to the OpenAI API, allowing us to utilize its services for tasks such as generating embeddings and interacting with language models.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The config dictionary is a structured configuration for initializing the Memory client. 
# It specifies the version of the configuration, the providers and settings for the embedder (used for generating vector embeddings),
# the language model (LLM) for processing queries, and the graph and vector stores for storing and retrieving data. 
# Each component is configured with necessary parameters such as API keys, model names, and connection details to external services like Neo4j and Qdrant.
config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": { "api_key": OPENAI_API_KEY, "model": "text-embedding-3-small" }
    },
    "llm": {
        "provider": "openai",
        "config": { "api_key": OPENAI_API_KEY, "model": "gpt-4.1" }
    },
    # The graph_store section of the configuration specifies the use of Neo4j as the graph database provider.
    # It includes the connection details such as the URL, username, and password required to connect to the Neo4j database.
    "graph_store":{
        "provider": "neo4j",
        "config": { 
            "url": "neo4j+s://fd10af2d.databases.neo4j.io",
            "username": "neo4j",
            "password": "Ri_hsFutNnQr0YyGeRzHroB_GlvYE-ATBVgMbu9Ohco"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    }
}

# This client serves as the interface for interacting with the memory system, 
# allowing for operations such as searching for relevant memories based on user queries and adding new memories to the system.
mem_client = Memory.from_config(config)


while True:


    user_query = input("> ")

#
    search_memory = mem_client.search(query=user_query, filters={"user_id": "piyushgarg"})


    """
    memories = []

    for mem in search_memory.get("results", []):

        formatted_memory = (
            f"ID: {mem.get('id')}\n"
            f"Memory: {mem.get('memory')}"
        )

        memories.append(formatted_memory) 
    """
    memories = [
        #
        f"ID: {mem.get('id')}\nMemory: {mem.get('memory')}" 
        for mem in search_memory.get("results")
    ]

    print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """
    print("json.dumps(memories)", json.dumps(memories))

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_query }
        ]
    )

    ai_response = response.choices[0].message.content

    print("AI:", ai_response)

    mem_client.add(
        user_id="piyushgarg",
        messages=[
            { "role": "user", "content": user_query },
            { "role": "assistant", "content": ai_response }
        ]
    )

    print("Memory has been saved...")