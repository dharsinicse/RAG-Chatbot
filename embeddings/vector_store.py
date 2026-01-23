import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def load_and_chunk_text(file_path):
    """
    Load raw text from file and split into overlapping chunks
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    return chunks


def save_index(index, chunks, index_path, chunks_path):
    """Saves the FAISS index and chunks to disk."""
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

def load_existing_index(index_path, chunks_path):
    """Loads a FAISS index and chunks from disk."""
    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        return None, None
    
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def create_faiss_index(chunks):
    """
    Create FAISS vector index from text chunks
    """
    print("🔹 Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("🔹 Creating embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks, model