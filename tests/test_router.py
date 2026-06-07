import os
import pytest
from agent.router import AgentRouter

def test_heuristic_routing():
    router = AgentRouter()
    
    # SQL query queries
    assert router.route("how many products do we have?") == "sql_query"
    assert router.route("what is the total sales revenue?") == "sql_query"
    assert router.route("who bought enterprise plans?") == "sql_query"
    
    # Web search queries
    assert router.route("what is the current stock price of Apple?") == "web_search"
    assert router.route("latest news about LLMs") == "web_search"
    
    # Direct answer queries
    assert router.route("hello") == "direct"
    assert router.route("hi there") == "direct"
    
    # Vector DB queries (default)
    assert router.route("what is the company refund policy?") == "vector_db"
    assert router.route("tell me about the security encryption standards") == "vector_db"
