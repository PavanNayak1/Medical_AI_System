import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdfs, filter_minimal, text_split, text_embedding

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

extract_pdfs = load_pdfs("/home/user/Desktop/medical_bot/Eye_Disease_Chatbot/Data")
filtered_docs = filter_minimal(extract_pdfs)
text_chunks = text_split(filtered_docs)
embeddings = text_embedding()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding=embeddings,
    index_name="eye-disease-chatbot"
)

