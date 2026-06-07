import streamlit as st
import requests
import json
import uuid
import os
from dotenv import load_dotenv

# Ensure page layout is wide and styled nicely
st.set_page_config(
    page_title="Agentic RAG Pipeline Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via injected CSS
st.markdown("""
<style>
    /* Main body background and font */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers custom styling */
    h1 {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.05em;
    }
    
    /* Sidebar styling adjustment */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Cards and boxes */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        backdrop-filter: blur(12px);
    }
    
    /* Chat area styling */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 16px;
        margin-bottom: 12px;
        line-height: 1.5;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .assistant-bubble {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Base URL for API
API_URL = "http://localhost:8000"

# Initialize Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# Sidebar Controls
st.sidebar.title("🛠️ Configuration Panel")
st.sidebar.markdown("Configure agent parameters and vector store details.")

# Environment status
st.sidebar.subheader("🔌 Connection Status")
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3).json()
    is_connected = True
    mock_active = health_resp.get("mock_mode", False)
    doc_count = health_resp.get("document_count", 0)
except Exception:
    is_connected = False
    mock_active = True
    doc_count = 0

if is_connected:
    st.sidebar.success("FastAPI Server: Connected")
    st.sidebar.caption(f"Indexed Chunks: **{doc_count}**")
else:
    st.sidebar.error("FastAPI Server: Disconnected")
    st.sidebar.info("Ensure the FastAPI API is running at localhost:8000. Operating in offline frontend-only fallback.")

# Global Mock Mode Toggle (Updates local environment if possible)
mock_mode = st.sidebar.toggle("Mock Mode (Offline Simulation)", value=mock_active)
if mock_mode != mock_active:
    # Update local .env or state
    os.environ["MOCK_MODE"] = str(mock_mode)

st.sidebar.divider()

# Hyperparameters Sliders
st.sidebar.subheader("📐 Hyperparameters")
chunk_size = st.sidebar.slider("Chunk Size (tokens)", min_value=128, max_value=1024, value=512, step=64)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=256, value=64, step=16)
top_k = st.sidebar.slider("Top K Retrieval", min_value=1, max_value=15, value=5)
mmr_lambda = st.sidebar.slider("MMR Lambda (Relevance vs Diversity)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
confidence_threshold = st.sidebar.slider("Grader Confidence Threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05)

st.sidebar.divider()

# Document Ingestion Helper
st.sidebar.subheader("📥 Document Ingestor")
ingest_path = st.sidebar.text_input("Local File or Directory Path:", value="./docs" if os.path.exists("./docs") else "")
if st.sidebar.button("Index Documents"):
    if not ingest_path:
        st.sidebar.warning("Please specify a path.")
    else:
        with st.sidebar.status("Ingesting documents..."):
            try:
                res = requests.post(f"{API_URL}/ingest", json={"file_path": ingest_path})
                if res.status_code == 200:
                    st.sidebar.success(f"Successfully indexed documents from: {ingest_path}!")
                else:
                    st.sidebar.error(f"Failed: {res.json().get('detail')}")
            except Exception as e:
                st.sidebar.error(f"Error connecting to backend: {e}")

# Header
st.title("🧠 Agentic RAG Pipeline Dashboard")
st.markdown("A production-grade Retrieval-Augmented Generation system with intelligent routing, multi-source retrieval, and quality assurance check.")

# Main grid layout
col_chat, col_routing = st.columns([2.2, 1])

# Left column: Chat History
with col_chat:
    st.markdown("### 💬 Interactive Assistant")
    
    # Display messages
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble assistant-bubble">{content}</div>', unsafe_allow_html=True)
            
    # Input area
    user_query = st.chat_input("Ask a question (e.g. 'What is the refund policy?', 'How many licenses did we sell?')")
    
    if user_query:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.markdown(f'<div class="chat-bubble user-bubble">{user_query}</div>', unsafe_allow_html=True)
        
        # Clear logs and sources
        st.session_state.pipeline_logs = ["Preprocessing query..."]
        st.session_state.sources = []
        
        # Assistant streaming target
        with st.markdown('<div class="chat-bubble assistant-bubble">', unsafe_allow_html=True):
            assistant_placeholder = st.empty()
            full_response = ""
            
            # Make API Call or generate fallback
            if is_connected and not mock_mode:
                try:
                    payload = {"message": user_query, "session_id": st.session_state.session_id}
                    headers = {"Content-Type": "application/json"}
                    
                    response = requests.post(f"{API_URL}/chat", json=payload, headers=headers, stream=True)
                    
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                try:
                                    data_json = json.loads(data_str)
                                    
                                    # Route updates
                                    if "routing_path" in data_json:
                                        st.session_state.pipeline_logs = data_json["routing_path"]
                                        
                                    # Tokens
                                    if "token" in data_json:
                                        full_response += data_json["token"]
                                        assistant_placeholder.markdown(full_response + "▌")
                                        
                                    # Completion and sources
                                    if data_json.get("done") is True:
                                        st.session_state.sources = data_json.get("sources", [])
                                except Exception:
                                    pass
                except Exception as e:
                    full_response = f"Network connection error: {e}. Fallback to simulated local generation."
                    st.session_state.pipeline_logs = ["Local Fallback Route: direct"]
                    st.session_state.sources = ["Local knowledge database"]
                    assistant_placeholder.markdown(full_response)
            else:
                # Local Simulation if not connected or mock is toggled
                st.session_state.pipeline_logs = [
                    "Query preprocessed: standalone form generated",
                    f"Heuristic Routing Selected: {'sql_query' if any(w in user_query.lower() for w in ['how many', 'sale', 'sum', 'revenue', 'db', 'sql']) else 'vector_db'}",
                    "Quality Grade: 0.85 (Context sufficient)"
                ]
                
                # Mock generation response
                from generation.generator import AnswerGenerator
                from retrieval.web_retriever import WebRetriever
                
                generator = AnswerGenerator()
                web = WebRetriever()
                
                route = "vector_db"
                if any(w in user_query.lower() for w in ["how many", "sale", "sum", "revenue", "db", "sql"]):
                    route = "sql_query"
                    context = '[{"sum(total_amount)": 16442.0, "count(*)": 6}]'
                    st.session_state.sources = ["SQL query database"]
                else:
                    context = web._generate_mock_results(user_query)[0].page_content
                    st.session_state.sources = [web._generate_mock_results(user_query)[0].metadata["source"]]
                
                for token in generator.generate_stream(user_query, context, route):
                    full_response += token
                    assistant_placeholder.markdown(full_response + "▌")
                    
            assistant_placeholder.markdown(full_response)
            
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# Right column: Routing Log & Citations
with col_routing:
    st.markdown("### 🔍 Agent Execution Tracing")
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("#### **🛠️ Decision Path & Routing Log**")
    if not st.session_state.pipeline_logs:
        st.info("Ask a question to see the agent's real-time reasoning trace.")
    else:
        for idx, log in enumerate(st.session_state.pipeline_logs):
            st.markdown(f"**Step {idx+1}:** {log}")
    st.markdown('</div><br>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("#### **📄 Retrieved Sources & Citations**")
    if not st.session_state.sources:
        st.info("No sources retrieved yet.")
    else:
        for src in st.session_state.sources:
            st.markdown(f'<span class="source-badge">{src}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
