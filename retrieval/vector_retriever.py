import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from ingestion.embedder import get_embeddings, get_vectorstore

class VectorRetriever:
    """Retrieves document chunks from vector store using MMR."""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = get_vectorstore(self.embeddings)
        self.top_k = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self.mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.7"))
        
    def retrieve(self, query: str) -> List[Document]:
        """Performs Maximal Marginal Relevance (MMR) search."""
        try:
            print(f"Retrieving from Vector DB: '{query}' (top_k={self.top_k}, lambda={self.mmr_lambda})")
            # MMR search balances relevance with diversity
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=self.top_k,
                fetch_k=20,
                lambda_mult=self.mmr_lambda
            )
            return docs
        except Exception as e:
            print(f"Error retrieving from vector store: {e}")
            return []
