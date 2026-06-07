import os
import pytest
import json
from retrieval.sql_retriever import SQLRetriever
from retrieval.web_retriever import WebRetriever

def test_sql_retriever_heuristics():
    retriever = SQLRetriever()
    
    # Test products query
    docs = retriever.retrieve("Show me all products")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "sql_database"
    assert "Enterprise Plan License" in docs[0].page_content
    
    # Test orders query
    docs_orders = retriever.retrieve("Show all orders")
    assert len(docs_orders) == 1
    # Check that it executed a valid SQL returning customer or order fields
    data = json.loads(docs_orders[0].page_content)
    assert len(data) > 0
    assert "customer_id" in data[0] or "customer_name" in data[0]
    
def test_web_retriever_mock():
    retriever = WebRetriever()
    
    # Check that it returns fallback documents in mock or live mode
    docs = retriever.retrieve("What is the latest software pricing?")
    assert len(docs) >= 1
    assert "pricing" in docs[0].metadata["source"] or "http" in docs[0].metadata["source"]
    assert "pricing" in docs[0].page_content.lower() or "plan" in docs[0].page_content.lower()
