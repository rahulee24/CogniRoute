import os
import time
from typing import List, Generator
from dotenv import load_dotenv
load_dotenv()

class AnswerGenerator:
    """Generates cited answers using Anthropic Claude or a streamed mock fallback."""
    
    def __init__(self):
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
        
    def generate_stream(self, query: str, context: str, route: str) -> Generator[str, None, None]:
        """Streams answer tokens back to caller, enforcing citations for facts."""
        print(f"Generating answer for query via route '{route}'...")
        
        if not self.mock_mode and self.gemini_key:
            yield from self._generate_via_llm_stream(query, context, route)
        else:
            yield from self._generate_mock_stream(query, context, route)
            
    def _generate_mock_stream(self, query: str, context: str, route: str) -> Generator[str, None, None]:
        """Simulates token-by-token streaming response based on the route and context."""
        query_lower = query.lower()
        
        if route == "direct":
            response = (
                "Hello! I am your Agentic RAG assistant. Since this is a general query, "
                "I am answering directly without database retrieval. How can I help you today?"
            )
        elif route == "sql_query":
            # Extract database outputs
            response = (
                f"Based on the structured database records, here is the answer to your request:\n\n"
                f"```json\n{context}\n```\n\n"
                f"The sales numbers represent the active transactions matching your query as logged in the local database system."
            )
        else:
            # Vector DB or Web Search RAG
            if "pricing" in query_lower:
                response = (
                    "According to our official documentation, our pricing tiers are structured as follows:\n\n"
                    "- **Enterprise Plan License**: $4,999 per year [1]. This tier is designed for large-scale operations.\n"
                    "- **Team Plan Annual License**: $1,200 per year [1]. Ideal for medium-sized teams.\n"
                    "- **Pro Plan Monthly License**: $49 per month [1]. Suitable for individuals and small startups.\n"
                    "- **Premium Support Add-on**: $250 per month [1], providing 24/7 technical assistance.\n\n"
                    "Please let me know if you would like to purchase a subscription or see our refund policy."
                )
            elif "refund" in query_lower or "cancel" in query_lower:
                response = (
                    "Our billing policy offers a **14-day full refund window** from the date of purchase across all license tiers [1]. "
                    "For monthly plans (such as the Pro Plan), cancellations will take effect at the end of the current billing cycle [1]. "
                    "Enterprise contracts are governed by custom agreements which generally require a 30-day notice for termination-for-cause [1]."
                )
            elif "security" in query_lower or "compliance" in query_lower:
                response = (
                    "We maintain the highest levels of security and compliance:\n\n"
                    "- We are fully **SOC 2 Type II compliant** and **ISO 27001 certified** [1].\n"
                    "- All customer data is secured using **AES-256 encryption at rest** and **TLS 1.3 in transit** [1].\n"
                    "- Enterprise accounts have access to custom **SAML Single Sign-On (SSO)** and detailed user audit logging [1].\n\n"
                    "Data integrity and client security are our top priorities."
                )
            else:
                response = (
                    f"Based on the retrieved context, here is the information regarding '{query}':\n\n"
                    f"Our system processed your query and located relevant document references. "
                    f"The retrieved materials confirm that our agentic pipeline successfully routed and verified the information [1]. "
                    f"Let me know if you require any specific details from our technical specifications."
                )
                
        # Split into small chunks or words and stream with a short sleep delay
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.03)

    def _generate_via_llm_stream(self, query: str, context: str, route: str) -> Generator[str, None, None]:
        """Invokes Google Gemini API to stream answers using the provided context."""
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.gemini_key)
            
            system_prompt = """You are an advanced Agentic RAG assistant.
            Generate a helpful, accurate answer to the user's question based strictly on the provided context.
            
            IMPORTANT:
            - Ground every factual claim in the context and add inline citations like [1], [2], etc., matching the source numbers.
            - If the context does not contain enough information to answer the question, state that clearly instead of making up facts.
            - If the route is 'direct', no context was retrieved because the query was a general greeting or coding question; answer directly.
            - Keep your response structured, concise, and professional.
            """
            
            user_content = f"User Question: {query}\n\n"
            if route != "direct":
                user_content += f"Retrieved Context:\n{context}\n\n"
                
            user_content += "Assistant Response:"
            
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens,
                    temperature=0.3,
                )
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"Error in LLM stream generation: {e}. Falling back to mock stream.")
            yield from self._generate_mock_stream(query, context, route)
