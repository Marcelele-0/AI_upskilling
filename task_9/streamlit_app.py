"""
Streamlit Frontend for RAG System
Interactive web interface for the Azure Functions-based RAG system.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import io
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title=os.getenv("APP_TITLE", "RAG System - Chat Interface"),
    page_icon=os.getenv("APP_ICON", "🤖"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and better styling
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #1e1e1e;
}

/* Chat message styling */
.chat-message {
    padding: 1rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.user-message {
    background-color: #1a365d;
    border-left: 4px solid #4299e1;
    color: #e2e8f0;
}
.assistant-message {
    background-color: #2d1b3d;
    border-left: 4px solid #9f7aea;
    color: #e2e8f0;
}

/* Source cards */
.source-card {
    background-color: #2d3748;
    padding: 0.8rem;
    border-radius: 0.5rem;
    margin-top: 0.5rem;
    border-left: 3px solid #ed8936;
    color: #e2e8f0;
    border: 1px solid #4a5568;
}

/* Metric cards */
.metric-card {
    background-color: #2d3748;
    padding: 1rem;
    border-radius: 0.8rem;
    text-align: center;
    border: 1px solid #4a5568;
    color: #e2e8f0;
}

/* Input styling */
.stTextInput > div > div > input {
    background-color: #2d3748;
    color: #e2e8f0;
    border: 1px solid #4a5568;
}

.stTextArea > div > div > textarea {
    background-color: #2d3748;
    color: #e2e8f0;
    border: 1px solid #4a5568;
}

/* Button styling */
.stButton > button {
    background-color: #4299e1;
    color: white;
    border: none;
    border-radius: 0.5rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #3182ce;
    box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
}

/* Form submit button */
.stFormSubmitButton > button {
    background-color: #48bb78;
    color: white;
    border: none;
    border-radius: 0.5rem;
    width: 100%;
}

.stFormSubmitButton > button:hover {
    background-color: #38a169;
}

/* Success/Error message styling */
.stSuccess {
    background-color: #2f855a;
    color: #f0fff4;
}

.stError {
    background-color: #e53e3e;
    color: #fed7d7;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #2d3748;
    border-radius: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    color: #a0aec0;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: #4299e1;
    color: white;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #2d3748;
    color: #e2e8f0;
}

/* JSON display styling */
.stJson {
    background-color: #1a202c;
    border: 1px solid #4a5568;
}
</style>
""", unsafe_allow_html=True)

class RAGFrontend:
    """Main class for the RAG System Frontend"""
    
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'azure_function_url' not in st.session_state:
            st.session_state.azure_function_url = os.getenv("AZURE_FUNCTION_URL", "http://localhost:7071/api")
        if 'health_status' not in st.session_state:
            st.session_state.health_status = None
        if 'last_health_check' not in st.session_state:
            st.session_state.last_health_check = None
        if 'dark_theme' not in st.session_state:
            st.session_state.dark_theme = True
            
    def check_health(self, base_url: str) -> Dict:
        """Check the health of the Azure Function"""
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
            
    def call_rag_function(self, question: str, base_url: str) -> Dict:
        """Call the RAG Azure Function"""
        try:
            payload = {"question": question}
            response = requests.post(
                f"{base_url}/rag", 
                json=payload, 
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
            
    def upload_document(self, file_content: str, filename: str, base_url: str) -> Dict:
        """Upload a new document to the system"""
        # Note: This would need a corresponding Azure Function endpoint for document upload
        # For now, we'll simulate the functionality and show how it would work
        try:
            payload = {
                "content": file_content,
                "filename": filename,
                "timestamp": datetime.now().isoformat()
            }
            
            # This endpoint would need to be implemented in your Azure Function
            response = requests.post(
                f"{base_url}/upload", 
                json=payload, 
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}

def main():
    """Main Streamlit application"""
    
    frontend = RAGFrontend()
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Azure Function URL configuration
        st.subheader("Azure Function Settings")
        azure_url = st.text_input(
            "Azure Function Base URL",
            value=st.session_state.azure_function_url,
            help="Base URL of your Azure Function (without /api/rag)"
        )
        st.session_state.azure_function_url = azure_url
        
        # Health check
        st.subheader("System Health")
        if st.button("🔍 Check Health"):
            with st.spinner("Checking system health..."):
                health_result = frontend.check_health(azure_url)
                st.session_state.health_status = health_result
                st.session_state.last_health_check = datetime.now()
        
        # Display health status
        if st.session_state.health_status:
            health_data = st.session_state.health_status
            if health_data["status"] == "healthy":
                st.success("✅ System Healthy")
                if "data" in health_data:
                    st.json(health_data["data"])
            else:
                st.error(f"❌ System Unhealthy: {health_data.get('error', 'Unknown error')}")
                
        if st.session_state.last_health_check:
            st.caption(f"Last check: {st.session_state.last_health_check.strftime('%H:%M:%S')}")
            
        # Clear chat history
        st.subheader("Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main content area
    st.title("🤖 RAG System - Interactive Chat")
    st.markdown("Ask questions and get answers from your knowledge base!")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Upload Documents", "📊 Analytics"])
    
    with tab1:
        # Chat interface
        st.subheader("Chat with RAG System")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                            for j, source in enumerate(message["sources"]):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {j+1}</strong> (Score: {source.get('score', 'N/A'):.3f})<br>
                                    <small>ID: {source.get('id', 'N/A')}</small><br>
                                    {source.get('content', '')[:300]}{'...' if len(source.get('content', '')) > 300 else ''}
                                </div>
                                """, unsafe_allow_html=True)
        
        # Question input
        with st.form("question_form"):
            question = st.text_area(
                "Your Question:",
                placeholder="Type your question here...",
                height=100
            )
            submitted = st.form_submit_button("🚀 Ask Question")
            
        if submitted and question.strip():
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now()
            })
            
            # Call RAG function
            with st.spinner("🔍 Searching for answers..."):
                result = frontend.call_rag_function(question, azure_url)
                
            if result["status"] == "success":
                rag_response = result["data"]
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": rag_response.get("answer", "No answer provided"),
                    "sources": rag_response.get("sources", []),
                    "timestamp": datetime.now()
                })
                
                st.success("✅ Answer generated successfully!")
            else:
                st.error(f"❌ Error: {result['error']}")
                
            st.rerun()
    
    with tab2:
        # Document upload interface
        st.subheader("📄 Upload New Documents")
        st.markdown("Add new documents to expand your knowledge base.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'md', 'py', 'json'],
            help="Upload text-based files to add to your knowledge base"
        )
        
        if uploaded_file is not None:
            # Read file content
            file_content = uploaded_file.read().decode('utf-8')
            
            # Preview the content
            st.subheader("Preview:")
            st.text_area("File Content", file_content, height=200, disabled=True)
            
            # Metadata
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filename", uploaded_file.name)
            with col2:
                st.metric("Size", f"{len(file_content):,} characters")
                
            # Upload button
            if st.button("📤 Upload Document"):
                st.warning("⚠️ Document upload endpoint not yet implemented in Azure Function")
                st.info("""
                To implement document upload, you would need to:
                1. Add an '/upload' endpoint to your Azure Function
                2. Process the document content (chunking, embedding generation)
                3. Index the document in Azure Cognitive Search
                """)
                
                # Simulate upload for demo purposes
                st.json({
                    "filename": uploaded_file.name,
                    "content_length": len(file_content),
                    "status": "ready_for_processing"
                })
        
        # Manual text input
        st.subheader("📝 Add Text Content")
        with st.form("text_upload_form"):
            manual_title = st.text_input("Document Title")
            manual_content = st.text_area("Document Content", height=200)
            
            if st.form_submit_button("📤 Add Content"):
                if manual_title and manual_content:
                    st.warning("⚠️ Document upload endpoint not yet implemented")
                    # Here you would call frontend.upload_document()
                else:
                    st.error("Please provide both title and content")
    
    with tab3:
        # Analytics and statistics
        st.subheader("📊 Chat Analytics")
        
        if st.session_state.chat_history:
            # Basic statistics
            total_messages = len(st.session_state.chat_history)
            user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", total_messages)
            with col2:
                st.metric("Questions Asked", user_messages)
            with col3:
                st.metric("Answers Provided", assistant_messages)
            
            # Recent activity
            st.subheader("Recent Questions")
            recent_questions = [
                m for m in st.session_state.chat_history[-10:] 
                if m["role"] == "user"
            ]
            
            if recent_questions:
                for i, q in enumerate(reversed(recent_questions), 1):
                    st.markdown(f"**{i}.** {q['content'][:100]}{'...' if len(q['content']) > 100 else ''}")
            
            # Export chat history
            st.subheader("Export Data")
            if st.button("📥 Export Chat History"):
                chat_data = pd.DataFrame(st.session_state.chat_history)
                csv = chat_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"rag_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No chat history yet. Start asking questions to see analytics!")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **RAG System Frontend** | Built with Streamlit | Connected to Azure Functions")

if __name__ == "__main__":
    main()
