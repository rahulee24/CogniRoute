import os
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

class AgentRouter:
    """Decides the best retrieval tool or action for a given query."""
    
    def __init__(self):
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        
    def route(self, query: str) -> str:
        """Determines destination route for the query.
        Returns one of: 'vector_db', 'web_search', 'sql_query', 'direct'
        """
        if not self.mock_mode and self.gemini_key:
            route_decision = self._route_via_llm(query)
        else:
            route_decision = self._route_via_heuristics(query)
            
        print(f"Routing Decision: '{route_decision}' for query: '{query}'")
        return route_decision
        
    def _route_via_heuristics(self, query: str) -> str:
        """Keyword heuristics-based routing decision."""
        q = query.lower()
        
        # SQL-focused triggers (aggregations, counts, database structures)
        sql_keywords = [
            "how many", "sum", "average", "total sales", "revenue", "sold", "purchased", 
            "database", "sql", "order count", "popular product", "who bought", "clients list",
            "customers in", "billing records", "orders"
        ]
        if any(kw in q for kw in sql_keywords):
            return "sql_query"
            
        # Web Search triggers (current, real-time, external, or generic topics)
        web_keywords = [
            "latest", "current", "weather", "news", "today", "live", "stock price", 
            "recent", "website", "internet", "online", "ceo of", "founder of", "happen"
        ]
        if any(kw in q for kw in web_keywords):
            return "web_search"
            
        # Direct Conversation triggers (greetings, off-topic chat, formatting requests)
        direct_keywords = [
            "hello", "hi", "hey", "how are you", "who are you", "tell me a joke", 
            "explain coding", "write a function", "help me write"
        ]
        if any(kw in q for kw in direct_keywords) and len(q.split()) <= 6:
            return "direct"
            
        # Default RAG target (internal policies, pricing details, product features)
        return "vector_db"

    def _route_via_llm(self, query: str) -> str:
        """Employs Google Gemini to determine routing destination."""
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.gemini_key)
            
            prompt = """You are an intelligent RAG routing agent. Your job is to analyze the user's query and route it to the single best resource.
            Select exactly one from this list:
            - 'vector_db': for queries about internal company documentation, software product features, user manuals, internal refund policy, or security specifications.
            - 'web_search': for current news, live web info, real-time events, public internet details, or topics outside the company.
            - 'sql_query': for questions requiring structured database calculations, totals, list of customers, product inventories, or sales metrics.
            - 'direct': for general greetings, chatting, or requests that do not require any external documentation or retrieval (e.g., 'hello', 'write a poem', 'explain recursion').

            Only return the exact label ('vector_db', 'web_search', 'sql_query', or 'direct'). Do not include markdown formatting or explanations.

            Question: {query}
            
            Route:"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt.format(query=query),
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=0.0,
                )
            )
            
            route = response.text.strip().lower()
            
            # Sanity check output
            valid_routes = ["vector_db", "web_search", "sql_query", "direct"]
            for valid in valid_routes:
                if valid in route:
                    return valid
                    
            return "vector_db"  # Fallback
        except Exception as e:
            print(f"Error in LLM routing: {e}. Falling back to heuristics.")
            return self._route_via_heuristics(query)
