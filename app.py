import os
import tempfile
from typing import List
import json
from datetime import datetime

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
import requests
import streamlit as st
from agno.agent import Agent
from agno.document import Document
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.utils.log import logger
from utils import (
    CUSTOM_CSS,
    about_widget,
    add_message,
    display_tool_calls,
    export_chat_history,
    rename_session_widget,
    session_selector_widget,
)

# Import Graph RAG
try:
    from agentic_rag import (
        get_agentic_rag_agent,
        get_graph_rag_statistics,
        test_graph_rag_setup,
        setup_neo4j_docker_instructions,
        NEO4J_AVAILABLE,
        GRAPH_RAG_AVAILABLE,
        ANTHROPIC_AVAILABLE,
        GOOGLE_AVAILABLE,
        GROQ_AVAILABLE
    )
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Graph RAG not available: {e}")
    st.stop()

nest_asyncio.apply()
st.set_page_config(
    page_title="Graph RAG Enterprise",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    st.rerun()


def get_reader(file_type: str):
    """Return appropriate reader based on file type."""
    readers = {
        "pdf": PDFReader(),
        "csv": CSVReader(),
        "txt": TextReader(),
    }
    return readers.get(file_type.lower(), None)


def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🕸️ System Status")
    
    agent = st.session_state.get("agentic_rag_agent")
    
    if agent:
        try:
            stats = get_graph_rag_statistics(agent)
            
            # Vector DB status
            vector_status = "✅" if stats.get("vector_db_enabled") else "❌"
            st.sidebar.write(f"{vector_status} **Vector Search**: {'Enabled' if stats.get('vector_db_enabled') else 'Disabled'}")
            
            # Graph DB status  
            graph_status = "✅" if stats.get("graph_db_enabled") else "❌"
            st.sidebar.write(f"{graph_status} **Knowledge Graph**: {'Enabled' if stats.get('graph_db_enabled') else 'Disabled'}")
            
            # Tools status
            tools_count = stats.get("tools_available", 0)
            st.sidebar.write(f"🛠️ **Tools Available**: {tools_count}")
            
            # Show graph statistics if available
            if stats.get("graph_db_enabled"):
                with st.sidebar.expander("📊 Graph Statistics", expanded=False):
                    st.write(f"**Entities**: {stats.get('graph_entities', 0)}")
                    st.write(f"**Relationships**: {stats.get('graph_relationships', 0)}")
            
        except Exception as e:
            st.sidebar.error(f"Error getting stats: {str(e)}")
    else:
        st.sidebar.info("Initialize agent to see status")


def display_setup_instructions():
    """Display setup instructions if needed"""
    if st.session_state.get("show_setup", False):
        st.markdown("### 🐳 Setup Instructions")
        
        setup_text = setup_neo4j_docker_instructions()
        st.markdown(setup_text)
        
        if st.button("Test Connection"):
            with st.spinner("Testing connections..."):
                test_results = test_graph_rag_setup()
                
                st.write("**Test Results:**")
                for key, value in test_results.items():
                    status = "✅" if value else "❌"
                    st.write(f"{status} {key.replace('_', ' ').title()}: {value}")
        
        if st.button("Close Instructions"):
            st.session_state["show_setup"] = False
            st.rerun()


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>🕸️ Graph RAG Enterprise</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Advanced RAG with Neo4j Knowledge Graphs powered by Agno</p>",
        unsafe_allow_html=True,
    )

    # Display setup instructions if needed
    display_setup_instructions()

    ####################################################################
    # Model selector with availability checks
    ####################################################################
    available_models = {"o3-mini": "openai:o3-mini", "gpt-4o": "openai:gpt-4o"}
    
    if GOOGLE_AVAILABLE:
        available_models["gemini-2.0-flash-exp"] = "google:gemini-2.0-flash-exp"
    if ANTHROPIC_AVAILABLE:
        available_models["claude-3-5-sonnet"] = "anthropic:claude-3-5-sonnet-20241022"
    if GROQ_AVAILABLE:
        available_models["llama-3.3-70b"] = "groq:llama-3.3-70b-versatile"
    
    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=list(available_models.keys()),
        index=0,
        key="model_selector",
    )
    model_id = available_models[selected_model]
    
    # Show missing dependencies
    missing_deps = []
    if not ANTHROPIC_AVAILABLE:
        missing_deps.append("anthropic")
    if not GOOGLE_AVAILABLE:
        missing_deps.append("google-genai")
    if not GROQ_AVAILABLE:
        missing_deps.append("groq")
    
    if missing_deps:
        with st.sidebar.expander("📦 Install More Models", expanded=False):
            st.markdown("**Install additional model providers:**")
            for dep in missing_deps:
                st.code(f"pip install {dep}")

    # Graph features toggle
    if GRAPH_RAG_AVAILABLE and NEO4J_AVAILABLE:
        enable_graph = st.sidebar.checkbox(
            "🕸️ Enable Knowledge Graph",
            value=True,
            help="Enable Neo4j knowledge graph features"
        )
    else:
        enable_graph = False
        if not NEO4J_AVAILABLE:
            st.sidebar.info("💡 Install Neo4j: `pip install neo4j`")
        if st.sidebar.button("📋 Setup Instructions"):
            st.session_state["show_setup"] = True

    ####################################################################
    # Initialize Agent
    ####################################################################
    agentic_rag_agent: Agent
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or st.session_state.get("current_model") != model_id
        or st.session_state.get("current_graph_setting") != enable_graph
    ):
        logger.info(f"---*--- Creating new Graph RAG Agent ---*---")
        
        try:
            with st.spinner("🚀 Initializing Graph RAG Agent..."):
                agentic_rag_agent = get_agentic_rag_agent(
                    model_id=model_id, 
                    enable_graph=enable_graph
                )
                
                st.session_state["agentic_rag_agent"] = agentic_rag_agent
                st.session_state["current_model"] = model_id
                st.session_state["current_graph_setting"] = enable_graph
                
                if enable_graph and hasattr(agentic_rag_agent, '_neo4j_client') and agentic_rag_agent._neo4j_client:
                    st.success("✅ Graph RAG Agent initialized with Neo4j!")
                else:
                    st.info("📚 Vector RAG Agent initialized")
                    
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            if "not available" in str(e).lower():
                st.info("💡 Install missing dependencies or check setup instructions.")
                st.session_state["show_setup"] = True
            return
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    ####################################################################
    # Display system status
    ####################################################################
    display_system_status()

    ####################################################################
    # Load Agent Session
    ####################################################################
    session_id_exists = (
        "agentic_rag_agent_session_id" in st.session_state
        and st.session_state["agentic_rag_agent_session_id"]
    )

    if not session_id_exists:
        try:
            st.session_state["agentic_rag_agent_session_id"] = (
                agentic_rag_agent.load_session()
            )
        except Exception as e:
            logger.error(f"Session load error: {str(e)}")
            st.warning("Could not create Agent session, is the database running?")

    ####################################################################
    # Load runs from memory
    ####################################################################
    agent_runs = []
    if hasattr(agentic_rag_agent, "memory") and agentic_rag_agent.memory is not None:
        agent_runs = agentic_rag_agent.memory.runs

    # Initialize messages if it doesn't exist yet
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Only populate messages from agent runs if we haven't already
    if len(st.session_state["messages"]) == 0 and len(agent_runs) > 0:
        logger.debug("Loading run history")
        for _run in agent_runs:
            if hasattr(_run, "message") and _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            if hasattr(_run, "response") and _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)

    if prompt := st.chat_input("💬 Ask me anything about your documents or explore the knowledge graph!"):
        add_message("user", prompt)

    ####################################################################
    # Document Management
    ####################################################################
    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = set()
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = set()

    st.sidebar.markdown("#### 📚 Document Management")
    
    # URL input
    input_url = st.sidebar.text_input("Add URL to Knowledge Base")
    if input_url and not prompt:
        if input_url not in st.session_state.loaded_urls:
            alert = st.sidebar.info("Processing URL...", icon="ℹ️")
            
            if input_url.lower().endswith(".pdf"):
                try:
                    response = requests.get(input_url, stream=True, verify=False)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name

                    reader = PDFReader()
                    docs: List[Document] = reader.read(tmp_path)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.sidebar.error(f"Error processing PDF: {str(e)}")
                    docs = []
            else:
                scraper = WebsiteReader(max_links=2, max_depth=1)
                docs: List[Document] = scraper.read(input_url)

            if docs:
                with st.spinner("📊 Processing and building knowledge graph..."):
                    agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_urls.add(input_url)
                st.sidebar.success("URL added to knowledge base")
                if enable_graph:
                    st.sidebar.success("🕸️ Knowledge graph updated!")
            else:
                st.sidebar.error("Could not process the provided URL")
            alert.empty()
        else:
            st.sidebar.info("URL already loaded")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Add a Document (.pdf, .csv, or .txt)", key="file_upload"
    )
    if uploaded_file and not prompt:
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in st.session_state.loaded_files:
            alert = st.sidebar.info("Processing document...", icon="ℹ️")
            file_type = uploaded_file.name.split(".")[-1].lower()
            reader = get_reader(file_type)
            if reader:
                docs = reader.read(uploaded_file)
                with st.spinner("📊 Processing and building knowledge graph..."):
                    agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_files.add(file_identifier)
                st.sidebar.success(f"{uploaded_file.name} added to knowledge base")
                if enable_graph:
                    st.sidebar.success("🕸️ Knowledge graph updated!")
            alert.empty()
        else:
            st.sidebar.info(f"{uploaded_file.name} already loaded")
    # Display loaded documents
    if st.session_state.loaded_files or st.session_state.loaded_urls:
        st.sidebar.markdown("#### 📄 Loaded Documents")
        
        if st.session_state.loaded_files:
            st.sidebar.markdown("**Files:**")
            for file_id in st.session_state.loaded_files:
                file_name = file_id.split("_")[0]
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"📄 {file_name}")
                with col2:
                    if st.button("🗑️", key=f"rm_f_{hash(file_id)}"):
                        st.session_state.loaded_files.remove(file_id)
                        st.rerun()
        
        if st.session_state.loaded_urls:
            st.sidebar.markdown("**URLs:**")
            for url in st.session_state.loaded_urls:
                display_url = url[:30] + "..." if len(url) > 30 else url
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"🔗 {display_url}")
                with col2:
                    if st.button("🗑️", key=f"rm_u_{hash(url)}"):
                        st.session_state.loaded_urls.remove(url)
                        st.rerun()
    # Clear knowledge base
    if st.sidebar.button("🧹 Clear All Knowledge"):
        with st.spinner("Clearing all knowledge..."):
            try:
                # Force delete Qdrant collection completely
                try:
                    import requests
                    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                    requests.delete(f"{qdrant_url}/collections/graph_rag_docs")
                except:
                    pass
                
                # Clear Neo4j completely
                if enable_graph and hasattr(agentic_rag_agent, '_neo4j_client') and agentic_rag_agent._neo4j_client:
                    try:
                        agentic_rag_agent._neo4j_client.run_cypher_query("MATCH (n) DETACH DELETE n")
                    except:
                        pass
                
                # Clear session state
                st.session_state.loaded_urls.clear()
                st.session_state.loaded_files.clear()
                
                # Force restart agent with clean databases
                st.session_state["agentic_rag_agent"] = None
                st.session_state["agentic_rag_agent_session_id"] = None
                st.session_state["messages"] = []
                
                st.sidebar.success("✅ All knowledge cleared!")
                st.success("🔄 Restarting with clean databases...")
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Clear failed: {str(e)}")
    ###############################################################
    # Sample Questions
    ###############################################################
    st.sidebar.markdown("#### ❓ Sample Questions")
    
    sample_questions = [
        ("📝 Summarize", "Summarize the documents and show key entities"),
        ("🕸️ Entity Map", "What entities are mentioned and how are they connected?"),
        ("🔍 Find Relations", "Explore relationships in the knowledge graph"),
        ("📊 Graph Stats", "Show me knowledge graph statistics")
    ]
    
    for title, question in sample_questions:
        if st.sidebar.button(title, key=f"sample_{hash(question)}"):
            add_message("user", question)

    ###############################################################
    # Utility buttons
    ###############################################################
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            restart_agent()
    with col2:
        if st.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="graph_rag_chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        ):
            st.sidebar.success("Chat history exported!")

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            tool_calls_container = st.empty()
            resp_container = st.empty()
            
            spinner_text = "🕸️ Thinking and exploring knowledge graph..." if enable_graph else "🤔 Thinking..."
            
            with st.spinner(spinner_text):
                response = ""
                try:
                    run_response = agentic_rag_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message("assistant", response, agentic_rag_agent.run_response.tools)
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)

    ####################################################################
    # Session Management
    ####################################################################
    session_selector_widget(agentic_rag_agent, model_id)
    rename_session_widget(agentic_rag_agent)

    ####################################################################
    # About section
    ####################################################################
    about_widget()


if __name__ == "__main__":
    main()