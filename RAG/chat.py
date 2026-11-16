from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

# this program is to demonstrate the RAG (Retrieval Augmented Generation) concept using Langchain and Qdrant vector database. The program will load a PDF file, split it into smaller chunks, create vector embeddings for each chunk, store them in a Qdrant vector database, and then allow the user to ask questions based on the content of the PDF file. The AI assistant will answer the user's query based on the context retrieved from the PDF file along with page contents and page number.

# Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

# Take user input
user_query = input("Ask something: ")

# Relevant chunks from the vector db
search_results = vector_db.similarity_search(query=user_query)


#formatted_results = []

# for result in search_results:

#     formatted_text = (
#         f"Page Content: {result.page_content}\n"
#         f"Page Number: {result.metadata['page_label']}\n"
#         f"File Location: {result.metadata['source']}"
#     )

#     formatted_results.append(formatted_text)
#Alternatively, you can use a list comprehension to achieve the same result in a more concise way:(as above)

#This Python statement is creating a single string called context by combining multiple document search results into one formatted text block.

#You'll commonly see this pattern in RAG (Retrieval-Augmented Generation) applications, where retrieved document chunks are passed to an LLM as context.
# Example of the formatted context string:
# search_results = [
#     {
#         "page_content": "RAG retrieves relevant documents.",
#         "metadata": {
#             "page_label": "5",
#             "source": "AI_Guide.pdf"
#         }
#     },
#     {
#         "page_content": "LLMs generate answers using retrieved context.",
#         "metadata": {
#             "page_label": "8",
#             "source": "AI_Guide.pdf"
#         }
#     }
# ]
context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results]) # what 
# Page Content: RAG retrieves relevant documents.
# Page Number: 5
# File Location: AI_Guide.pdf

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
        { "role": "user", "content":user_query  },
    ]
)

print(f"🤖: {response.choices[0].message.content}")