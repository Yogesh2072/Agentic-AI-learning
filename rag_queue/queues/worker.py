from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

openai_client = OpenAI()

# Vector Embeddings
# it defines the embedding model to be used for generating vector representations of text data. The model specified here is "text-embedding-3-large", which is a large-scale embedding model provided by OpenAI. This model is designed to convert text into high-dimensional vectors that capture semantic meaning, allowing for tasks such as similarity search, clustering, and classification in natural language processing applications.
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

#it initializes a connection to an existing Qdrant vector database collection named "learning_rag" hosted at "http://localhost:6333". The QdrantVectorStore is configured to use the previously defined embedding model for generating vector representations of documents. This setup allows for efficient storage and retrieval of vectorized data, enabling similarity searches and other operations on the stored documents.
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

# it defines an asynchronous function named process_query that takes a user query as input. The function performs a similarity search on the vector database to retrieve relevant document chunks based on the query. It then constructs a context string containing the page content, page number, and file location of the retrieved documents. A system prompt is created to instruct the AI assistant to answer the user's query based on this context. Finally, the function sends the query and context to the OpenAI API for generating a response, which is printed and returned.
async def process_query(query:str):
    print("Searching Chunks", query)
    search_results = vector_db.similarity_search(query=query)

    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

    SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user to open the right page number to know more.

    Context:
    {context}
    """

    response = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            { "role": "system", "content":SYSTEM_PROMPT  },
            { "role": "user", "content":query  },
        ]
    )

    print(f"🤖: {response.choices[0].message.content}")
    return response.choices[0].message.content


