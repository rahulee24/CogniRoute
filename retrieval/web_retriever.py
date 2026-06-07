import os
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

class WebRetriever:
    """Retrieves live web results using Tavily Search API or mock fallback."""
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        
    def retrieve(self, query: str) -> List[Document]:
        """Queries Tavily API or returns mock results in offline/mock mode."""
        print(f"Retrieving from Web Search: '{query}'")
        
        if not self.mock_mode and self.api_key:
            try:
                from tavily import TavilyClient
                tavily = TavilyClient(api_key=self.api_key)
                response = tavily.search(query=query, max_results=5)
                
                documents = []
                for result in response.get("results", []):
                    documents.append(
                        Document(
                            page_content=result.get("content", ""),
                            metadata={
                                "source": result.get("url", "https://example.com"),
                                "title": result.get("title", "Web Page"),
                                "score": result.get("score", 0.0)
                            }
                        )
                    )
                return documents
            except Exception as e:
                print(f"Error querying Tavily: {e}. Falling back to mock results.")
                
        # Mock Results Fallback
        return self._generate_mock_results(query)
        
    def _generate_mock_results(self, query: str) -> List[Document]:
        """Generates realistic mock search results based on the query words."""
        query_lower = query.lower()
        
        if "pricing" in query_lower:
            content = "The enterprise plan is priced at $4,999/year. The annual team plan license costs $1,200/year, and the standard pro monthly subscription is $49/month. A 24/7 premium support add-on is available for $250/month. Custom volume discounts are available upon contacting sales."
            url = "https://example.com/pricing"
            title = "Official Software Pricing & License Tiers"
        elif "refund" in query_lower or "cancellation" in query_lower:
            content = "Customers can request a full refund within 14 days of purchase for any tier. For monthly plans, cancellations take effect at the end of the billing cycle. Enterprise agreements have a custom termination-for-cause clause with 30-day notice."
            url = "https://example.com/refund-policy"
            title = "Refund and Billing Policy Agreement"
        elif "security" in query_lower or "compliance" in query_lower:
            content = "Our services are fully SOC2 Type II compliant and ISO 27001 certified. We enforce AES-256 encryption at rest and TLS 1.3 in transit. Enterprise accounts feature custom SAML SSO and detailed user audit logging."
            url = "https://example.com/security"
            title = "Security Compliance and Encryption Standards"
        else:
            content = f"Search result for '{query}': Our platform provides state-of-the-art agentic workflows integrated with multi-source retrieval databases. Contact our customer support team for detailed whitepapers and documentation."
            url = "https://example.com/search-result"
            title = "Documentation Search Helpdesk"
            
        return [
            Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title,
                    "score": 0.95
                }
            )
        ]
