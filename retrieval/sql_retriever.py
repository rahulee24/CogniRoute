import os
import sqlite3
import json
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

class SQLRetriever:
    """Translates natural language queries to SQL, executes them on SQLite database, and returns results."""
    
    def __init__(self):
        self.db_path = os.getenv("SQLITE_DB_PATH", "data/company_sales.db")
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        
    def retrieve(self, query: str) -> List[Document]:
        """Runs natural language to SQL flow, executes query, and returns results as a Document."""
        print(f"Retrieving from SQL DB: '{query}'")
        
        sql_query = ""
        
        if not self.mock_mode and self.gemini_key:
            # Real LLM SQL Query generation
            sql_query = self._generate_sql_via_llm(query)
        else:
            # Rule-based / heuristics SQL generation (mock/fallback mode)
            sql_query = self._generate_sql_heuristics(query)
            
        print(f"Generated SQL Query: {sql_query}")
        
        if not sql_query:
            return [Document(
                page_content="No suitable SQL query could be generated to answer the database question.",
                metadata={"source": "sql_database", "query": sql_query}
            )]
            
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                content = "Query executed successfully but returned 0 results."
            else:
                # Convert rows to a list of dicts and then formatted JSON
                data = [dict(row) for row in rows]
                content = json.dumps(data, indent=2)
                
            return [Document(
                page_content=content,
                metadata={
                    "source": "sql_database",
                    "sql_query": sql_query,
                    "row_count": len(rows) if rows else 0
                }
            )]
            
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return [Document(
                page_content=f"Database execution error: {str(e)}",
                metadata={"source": "sql_database", "sql_query": sql_query, "error": str(e)}
            )]
            
    def _generate_sql_heuristics(self, query: str) -> str:
        """Heuristics-based Natural Language to SQL converter."""
        query_lower = query.lower()
        
        # Schema details:
        # products: product_id, name, category, price, stock_quantity
        # customers: customer_id, name, email, country
        # orders: order_id, customer_id, product_id, order_date, quantity, total_amount
        
        if "product" in query_lower or "what do we sell" in query_lower or "price" in query_lower:
            if "category" in query_lower or "type" in query_lower:
                return "SELECT DISTINCT category, COUNT(*), AVG(price) FROM products GROUP BY category"
            return "SELECT * FROM products"
            
        elif "customer" in query_lower or "who bought" in query_lower or "clients" in query_lower:
            if "country" in query_lower:
                return "SELECT country, COUNT(*) FROM customers GROUP BY country"
            return "SELECT * FROM customers"
            
        elif "order" in query_lower or "sales" in query_lower or "revenue" in query_lower or "sold" in query_lower:
            if "total" in query_lower or "revenue" in query_lower or "sum" in query_lower:
                return "SELECT SUM(total_amount) AS total_revenue, COUNT(*) AS total_orders FROM orders"
            if "popular" in query_lower or "best" in query_lower:
                return """
                    SELECT p.name, SUM(o.quantity) AS total_qty_sold 
                    FROM orders o 
                    JOIN products p ON o.product_id = p.product_id 
                    GROUP BY o.product_id 
                    ORDER BY total_qty_sold DESC 
                    LIMIT 1
                """
            return """
                SELECT o.order_id, c.name AS customer_name, p.name AS product_name, o.order_date, o.quantity, o.total_amount 
                FROM orders o 
                JOIN customers c ON o.customer_id = c.customer_id 
                JOIN products p ON o.product_id = p.product_id
            """
            
        # Default query
        return "SELECT * FROM products"
        
    def _generate_sql_via_llm(self, query: str) -> str:
        """Calls Google Gemini to translate query to SQLite SQL."""
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.gemini_key)
            
            schema = """
            Table structures:
            1. Table 'products':
               - product_id (INTEGER PRIMARY KEY)
               - name (TEXT)
               - category (TEXT)
               - price (REAL)
               - stock_quantity (INTEGER)
            2. Table 'customers':
               - customer_id (INTEGER PRIMARY KEY)
               - name (TEXT)
               - email (TEXT)
               - country (TEXT)
            3. Table 'orders':
               - order_id (INTEGER PRIMARY KEY)
               - customer_id (INTEGER, FOREIGN KEY references customers)
               - product_id (INTEGER, FOREIGN KEY references products)
               - order_date (TEXT, format 'YYYY-MM-DD')
               - quantity (INTEGER)
               - total_amount (REAL)
            """
            
            prompt = f"""You are a database expert. Translate this user question into a standard SQLite SELECT query.
            Only return the SQL query code block, starting with SELECT. Do not include any explanation or markdown backticks.
            
            Schema:
            {schema}
            
            Question: {query}
            
            SQL:"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.0,
                )
            )
            
            sql = response.text.strip()
            # Clean up SQL formatting if model included backticks
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "").replace("```", "").strip()
            elif sql.startswith("```"):
                sql = sql.replace("```", "").strip()
            return sql
        except Exception as e:
            print(f"Error in LLM SQL generation: {e}. Falling back to heuristics.")
            return self._generate_sql_heuristics(query)
