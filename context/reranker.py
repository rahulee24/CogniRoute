import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

class ContextReranker:
    """Reranks retrieved documents using Cohere Cross-Encoder or a local similarity fallback."""
    
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Reranks a list of Documents against the query."""
        if not documents:
            return []
            
        print(f"Reranking {len(documents)} document chunks...")
        
        if not self.mock_mode and self.api_key:
            try:
                import cohere
                co = cohere.Client(api_key=self.api_key)
                
                # Format texts for Cohere API
                doc_contents = [doc.page_content for doc in documents]
                
                # Call Cohere Rerank
                response = co.rerank(
                    query=query,
                    documents=doc_contents,
                    top_n=len(documents),
                    model="rerank-english-v3.0"
                )
                
                reranked_docs = []
                for result in response.results:
                    idx = result.index
                    doc = documents[idx]
                    # Update metadata with rerank score
                    doc.metadata["rerank_score"] = float(result.relevance_score)
                    reranked_docs.append(doc)
                    
                return reranked_docs
            except Exception as e:
                print(f"Error during Cohere reranking: {e}. Using fallback similarity scorer.")
                
        # Heuristics/TF-IDF similarity fallback
        return self._fallback_rerank(query, documents)
        
    def _fallback_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Simple overlap/relevance score calculation as a fallback."""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            content_words = doc.page_content.lower().split()
            # Simple match score
            matches = sum(1 for word in query_words if word in content_words)
            score = matches / max(len(query_words), 1)
            
            # Save score
            doc.metadata["rerank_score"] = score
            scored_docs.append((score, doc))
            
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]
