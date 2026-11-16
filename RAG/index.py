from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
# this program is to demonstrate the RAG (Retrieval Augmented Generation) concept using Langchain and Qdrant vector database. The program will load a PDF file, split it into smaller chunks, create vector embeddings for each chunk, store them in a Qdrant vector database, and then allow the user to ask questions based on the content of the PDF file. The AI assistant will answer the user's query based on the context retrieved from the PDF file along with page contents and page number.

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load this file in python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print(docs[12])

#Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400 #chunks will have 400 characters of overlap with the previous chunk
)

chunks = text_splitter.split_documents(documents=docs)

# # Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("Indexing of documents done....")