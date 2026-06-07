import os
from typing import List
from langchain_core.documents import Document

class ContextAssembler:
    """Assembles and formats text context for LLM consumption within a token budget."""
    
    def __init__(self, token_budget: int = 3000):
        self.token_budget = token_budget
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
            
    def count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken, falling back to approximation (char count / 4)."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4
        
    def assemble(self, documents: List[Document]) -> str:
        """Formats and fits documents into a single context string within the budget."""
        formatted_chunks = []
        current_tokens = 0
        
        for idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown Source")
            # If it's a file path, extract the basename for readability
            if os.path.exists(source) or "/" in source or "\\" in source:
                source = os.path.basename(source)
                
            page = doc.metadata.get("page")
            page_str = f" (Page {page})" if page else ""
            
            chunk_header = f"--- SOURCE [{idx+1}]: {source}{page_str} ---\n"
            chunk_body = f"{doc.page_content.strip()}\n\n"
            chunk_text = chunk_header + chunk_body
            
            chunk_tokens = self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > self.token_budget:
                print(f"Token budget reached. Stopping context assembly at {idx} documents.")
                break
                
            formatted_chunks.append(chunk_text)
            current_tokens += chunk_tokens
            
        return "".join(formatted_chunks).strip()
