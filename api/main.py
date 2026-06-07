import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Import pipeline components
from agent.preprocessor import QueryPreprocessor
from agent.router import AgentRouter
from agent.memory import SessionMemory
from retrieval.vector_retriever import VectorRetriever
from retrieval.web_retriever import WebRetriever
from retrieval.sql_retriever import SQLRetriever
from context.reranker import ContextReranker
from context.assembler import ContextAssembler
from quality.grader import ContextGrader
from generation.generator import AnswerGenerator
from ingestion.embedder import get_embeddings, get_vectorstore, ingest_documents

app = FastAPI(title="Agentic RAG Pipeline API", version="1.0.0")

# Initialize shared components
preprocessor = QueryPreprocessor()
router = AgentRouter()
memory = SessionMemory()
vector_retriever = VectorRetriever()
web_retriever = WebRetriever()
sql_retriever = SQLRetriever()
reranker = ContextReranker()
assembler = ContextAssembler(token_budget=3000)
grader = ContextGrader()
generator = AnswerGenerator()

class ChatRequest(BaseModel):
    message: str
    session_id: str

class IngestRequest(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, Any]] = None

@app.get("/health")
def health_check():
    """Health check endpoint returning DB state and vectorstore status."""
    try:
        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)
        # Check document count if Chroma is available
        doc_count = 0
        if hasattr(vectorstore, "_collection") and vectorstore._collection is not None:
            doc_count = vectorstore._collection.count()
            
        return {
            "status": "healthy",
            "mock_mode": os.getenv("MOCK_MODE", "False").lower() == "true",
            "vector_store": "chroma",
            "document_count": doc_count,
            "sql_db_path": os.getenv("SQLITE_DB_PATH", "data/company_sales.db")
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/ingest")
def ingest_file(req: IngestRequest):
    """Ingests a local file or directory into the vector store."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File/directory not found: {req.file_path}")
        
    try:
        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)
        ingest_documents(req.file_path, vectorstore)
        return {"status": "success", "message": f"Successfully ingested {req.file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Processes message through the Agentic RAG pipeline and streams response tokens."""
    
    def event_stream():
        session_id = req.session_id
        original_query = req.message
        
        # 1. Retrieve history
        history = memory.get_history(session_id)
        
        # 2. Preprocess & rewrite query
        query = preprocessor.preprocess(original_query, history)
        print(f"Preprocessed query: '{query}'")
        
        # 3. Router decision
        selected_route = router.route(query)
        
        # Keep track of sources and routing logs
        sources = []
        routing_log = [f"Initial route: {selected_route}"]
        
        max_reroutes = int(os.getenv("MAX_REROUTES", "2"))
        reroute_count = 0
        retrieved_docs = []
        context_str = ""
        
        # RAG Iterative retrieval loop with quality grader
        while reroute_count <= max_reroutes:
            # 4. Perform retrieval based on selected route
            if selected_route == "vector_db":
                retrieved_docs = vector_retriever.retrieve(query)
                # Apply reranking
                retrieved_docs = reranker.rerank(query, retrieved_docs)
                context_str = assembler.assemble(retrieved_docs)
                
                # Extract file sources
                sources = list(set([doc.metadata.get("source", "doc") for doc in retrieved_docs]))
                
                # Grade quality of context
                grade_result = grader.grade(query, context_str)
                print(f"Vector DB Grading: {grade_result}")
                
                if grade_result["sufficient"] or reroute_count == max_reroutes:
                    routing_log.append("Vector DB context accepted")
                    break
                else:
                    # Low quality context -> trigger re-routing to Web Search
                    selected_route = "web_search"
                    reroute_count += 1
                    routing_log.append(f"Low confidence ({grade_result['score']:.2f}). Re-routing to {selected_route} (attempt {reroute_count})")
                    
            elif selected_route == "web_search":
                retrieved_docs = web_retriever.retrieve(query)
                # Apply reranking
                retrieved_docs = reranker.rerank(query, retrieved_docs)
                context_str = assembler.assemble(retrieved_docs)
                
                # Extract URL sources
                sources = list(set([doc.metadata.get("source", "web") for doc in retrieved_docs]))
                
                # Grade quality of context
                grade_result = grader.grade(query, context_str)
                print(f"Web Search Grading: {grade_result}")
                
                if grade_result["sufficient"] or reroute_count == max_reroutes:
                    routing_log.append("Web context accepted")
                    break
                else:
                    # Fallback to direct answering if web retrieval also fails to be graded sufficient
                    selected_route = "direct"
                    reroute_count += 1
                    routing_log.append(f"Low confidence ({grade_result['score']:.2f}). Re-routing to {selected_route} (attempt {reroute_count})")
                    
            elif selected_route == "sql_query":
                # SQL Retriever executes and yields formatted query string
                sql_docs = sql_retriever.retrieve(query)
                context_str = sql_docs[0].page_content
                sql_query = sql_docs[0].metadata.get("sql_query", "")
                sources = [f"SQL Database Query: {sql_query}"] if sql_query else ["SQL Database"]
                routing_log.append("SQL query execution complete")
                break
                
            elif selected_route == "direct":
                context_str = ""
                sources = ["Direct LLM Knowledge"]
                routing_log.append("Direct answer generation")
                break
        
        # Broadcast initial metadata event (routing path)
        yield f"data: {json.dumps({'routing_path': routing_log})}\n\n"
        
        # 5. Generate and stream answer
        full_answer = ""
        try:
            for token in generator.generate_stream(query, context_str, selected_route):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            error_msg = f"Generation error: {e}"
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            full_answer += error_msg
            
        # Clean up citations formatting for sources list (e.g. keep basenames)
        clean_sources = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://") or src.startswith("SQL") or src.startswith("Direct"):
                clean_sources.append(src)
            else:
                clean_sources.append(os.path.basename(src))
                
        # 6. Save memory history
        memory.add_message(session_id, "user", original_query)
        memory.add_message(session_id, "assistant", full_answer)
        
        # Final Event
        yield f"data: {json.dumps({'done': True, 'sources': clean_sources})}\n\n"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")
