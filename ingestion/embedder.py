import os
import argparse
from typing import List
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker

class GeminiEmbeddings(Embeddings):
    """Google Gemini Embeddings implementation using the google-genai client."""
    def __init__(self, gemini_key: str):
        self.gemini_key = gemini_key
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            print(f"Error generating Gemini embeddings: {e}")
            # Fallback outputting 3072 dimensions
            return [[0.0] * 3072 for _ in texts]
            
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Simple deterministic Mock Embeddings for offline testing and fallback (3072 dims for Gemini)
class MockEmbeddings(Embeddings):
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import random
        embeddings = []
        for text in texts:
            # Deterministic seeding using hash
            h = hash(text)
            random.seed(h)
            vec = [random.gauss(0, 1) for _ in range(self.dimension)]
            norm = sum(x**2 for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    """Returns GeminiEmbeddings if API key is present and MOCK_MODE is False.
    Otherwise, returns MockEmbeddings.
    """
    mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    if not mock_mode and gemini_key:
        print("Using GeminiEmbeddings...")
        return GeminiEmbeddings(gemini_key=gemini_key)
    else:
        print("Using MockEmbeddings (offline mode)...")
        return MockEmbeddings()


def get_vectorstore(embeddings: Embeddings) -> Chroma:
    """Initializes and returns a local Chroma persistent vector store."""
    persist_directory = "data/chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    return Chroma(
        collection_name="agentic_rag_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


def ingest_documents(source_path: str, vector_store: Chroma):
    """Loads, chunks, and indexes documents from a source path."""
    print(f"Loading documents from: {source_path}")
    if os.path.isdir(source_path):
        documents = DocumentLoader.load_directory(source_path)
    else:
        documents = DocumentLoader.load_file(source_path)
        
    if not documents:
        print("No documents found to ingest.")
        return
        
    print(f"Loaded {len(documents)} document pages/sources.")
    
    # Retrieve configuration thresholds
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
    
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_documents(documents)
    print(f"Split into {len(chunks)} text chunks.")
    
    print("Uploading and indexing chunks in vector store...")
    vector_store.add_documents(chunks)
    print("Ingestion and indexing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--source", type=str, required=True, help="Path to file or directory to ingest")
    parser.add_argument("--vectorstore", type=str, default="chroma", choices=["chroma", "pinecone"], help="Vector store back-end")
    
    args = parser.parse_args()
    
    embeddings = get_embeddings()
    if args.vectorstore == "chroma":
        v_store = get_vectorstore(embeddings)
        ingest_documents(args.source, v_store)
    else:
        # Pinecone support
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("Error: PINECONE_API_KEY environment variable is missing. Defaulting to Chroma.")
            v_store = get_vectorstore(embeddings)
            ingest_documents(args.source, v_store)
        else:
            print("Pinecone is not fully configured locally. Using Chroma as fallback.")
            v_store = get_vectorstore(embeddings)
            ingest_documents(args.source, v_store)
