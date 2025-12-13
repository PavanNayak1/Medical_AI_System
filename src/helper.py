from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import tensorflow as tf
import numpy as np
import os
import re

#extracting text from pdf files
def load_pdfs(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def clean_text(docs: List[Document]) -> List[Document]:
    """
    Clean page_content for all documents in the list.
    """
    cleaned_docs = []
    for doc in docs:
        # Remove non-ASCII characters and unwanted blocks
        cleaned = re.sub(r'[^\x00-\x7F]+', '', doc.page_content)  # Remove non-ASCII
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize whitespace
        if cleaned:  # Skip empty after cleaning
            cleaned_docs.append(Document(page_content=cleaned, metadata=doc.metadata))
    return cleaned_docs

#filtering out empty documents

def filter_minimal(docs: List[Document])-> List[Document]:
    """
    Given a list of Documents objects, retuern a new list of Document objects
    containing only {source} and {page_content} attributes.
    """
    filtered : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        filtered.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    # Apply to all filtered_docs
    filtered_docs = clean_text(filtered)
    return filtered_docs

#splitting documents into smaller chunks
def text_split(cleaned_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=45,
        chunk_overlap=12
    )
    text_chunks = text_splitter.split_documents(cleaned_docs)
    return text_chunks


def text_embedding():
    """
    Download and Return HuggingFace Embeddings model.
    """
    Model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=Model_name

        )
    return embeddings

