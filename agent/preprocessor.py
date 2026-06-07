import os
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

class QueryPreprocessor:
    """Preprocesses queries by normalizing them and rewriting follow-up queries with chat history."""
    
    def __init__(self):
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        
    def preprocess(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Cleans query and merges history to output a standalone query."""
        cleaned_query = query.strip()
        
        if not chat_history or len(chat_history) == 0:
            return cleaned_query
            
        if not self.mock_mode and self.gemini_key:
            return self._rewrite_via_llm(cleaned_query, chat_history)
        else:
            return self._rewrite_via_heuristics(cleaned_query, chat_history)
            
    def _rewrite_via_heuristics(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Rule-based query rewriter for mock/offline use."""
        query_lower = query.lower()
        
        # Simple resolution: find the last mentioned key topics in history
        last_user_msg = ""
        last_assistant_msg = ""
        
        for msg in reversed(chat_history):
            if msg.get("role") == "user" and not last_user_msg:
                last_user_msg = msg.get("content", "")
            elif msg.get("role") == "assistant" and not last_assistant_msg:
                last_assistant_msg = msg.get("content", "")
                
        # Heuristics: if query contains pronouns like "it", "they", "pricing", "this", "that"
        context_words = []
        for text in [last_user_msg, last_assistant_msg]:
            for keyword in ["enterprise plan", "team plan", "pro plan", "refund", "security", "sales"]:
                if keyword in text.lower():
                    context_words.append(keyword)
                    break
                    
        if context_words:
            topic = context_words[0]
            # If the user is asking a short question, append the topic
            if len(query.split()) <= 4 or any(p in query_lower for p in ["it", "price", "cost", "refund", "buy", "cancel"]):
                return f"{query} regarding the {topic}"
                
        return query
        
    def _rewrite_via_llm(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Calls Google Gemini to resolve pronoun references and generate a standalone search query."""
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.gemini_key)
            
            history_str = ""
            for msg in chat_history[-5:]: # Look at last 5 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_str += f"{role}: {msg.get('content')}\n"
                
            prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone, search-friendly question (in English). 
            Do NOT answer the question. Only return the rewritten question.

            Conversation History:
            {history_str}
            
            Follow-up Question: {query}
            
            Rewritten standalone question:"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.1,
                )
            )
            
            return response.text.strip()
        except Exception as e:
            print(f"Error in LLM query preprocessing: {e}. Falling back to heuristics.")
            return self._rewrite_via_heuristics(query, chat_history)
