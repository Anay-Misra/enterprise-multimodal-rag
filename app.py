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

# Import Multimodal Support
try:
    from multimodal_readers import (
        MultimodalReader, ImageReader, AudioReader, VideoReader,
        get_reader_for_file, read_image, read_audio, read_video
    )
    from multimodal_processors import get_missing_dependencies
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multimodal support not available: {e}")
    MULTIMODAL_AVAILABLE = False

# Import Knowledge Validator
try:
    from knowledge_validator import add_knowledge_validator_to_agent, KnowledgeBaseValidator
    VALIDATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Knowledge validator not available: {e}")
    VALIDATOR_AVAILABLE = True

nest_asyncio.apply()
st.set_page_config(
    page_title="Multimodal Graph RAG Enterprise",
    page_icon="🎭",
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


def get_reader(file_type: str, file_name: str = ""):
    """Return appropriate reader based on file type."""
    
    # Check if it's a multimodal file first
    if MULTIMODAL_AVAILABLE:
        multimodal_extensions = {
            # Images
            'jpg': 'image', 'jpeg': 'image', 'png': 'image', 'bmp': 'image', 
            'tiff': 'image', 'gif': 'image',
            # Audio
            'mp3': 'audio', 'wav': 'audio', 'm4a': 'audio', 'flac': 'audio', 'ogg': 'audio',
            # Video
            'mp4': 'video', 'avi': 'video', 'mov': 'video', 'mkv': 'video', 'wmv': 'video'
        }
        
        if file_type.lower() in multimodal_extensions:
            modality = multimodal_extensions[file_type.lower()]
            
            if modality == 'image':
                return ImageReader(use_vision_model=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
            elif modality == 'audio':
                return AudioReader(openai_api_key=os.getenv("OPENAI_API_KEY"))
            elif modality == 'video':
                return VideoReader(extract_frames=5, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Standard readers for text documents
    readers = {
        "pdf": PDFReader(),
        "csv": CSVReader(),
        "txt": TextReader(),
    }
    return readers.get(file_type.lower(), None)


def display_knowledge_base_status():
    """Display what's actually in the knowledge base"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📚 Knowledge Base Status")
    
    agent = st.session_state.get("agentic_rag_agent")
    
    if agent and hasattr(agent, 'knowledge') and agent.knowledge:
        try:
            # Quick check of knowledge base
            if hasattr(agent.knowledge, 'vector_db') and agent.knowledge.vector_db:
                try:
                    # Try to get basic collection info
                    vector_db = agent.knowledge.vector_db
                    collection_name = getattr(vector_db, 'collection', 'default')
                    
                    if hasattr(vector_db, 'client') and vector_db.client:
                        try:
                            collection_info = vector_db.client.get_collection(collection_name)
                            doc_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                            
                            st.sidebar.write(f"📄 **Documents**: {doc_count}")
                            
                            if doc_count == 0:
                                st.sidebar.warning("⚠️ Knowledge base is empty")
                                st.sidebar.info("Upload documents to get started")
                            else:
                                st.sidebar.success(f"✅ {doc_count} documents loaded")
                            
                        except Exception as e:
                            st.sidebar.error(f"❌ Cannot access collection: {str(e)}")
                    else:
                        st.sidebar.warning("⚠️ Vector database not connected")
                except Exception as e:
                    st.sidebar.error(f"❌ Knowledge base error: {str(e)}")
            else:
                st.sidebar.warning("⚠️ No knowledge base configured")
        except Exception as e:
            st.sidebar.error(f"Error checking knowledge base: {str(e)}")
    else:
        st.sidebar.info("Initialize agent to see knowledge base status")


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
            
            # Multimodal status
            mm_status = "✅" if MULTIMODAL_AVAILABLE else "❌"
            st.sidebar.write(f"{mm_status} **Multimodal**: {'Enabled' if MULTIMODAL_AVAILABLE else 'Disabled'}")
            
            # Validation status
            val_status = "✅" if VALIDATOR_AVAILABLE else "❌"
            st.sidebar.write(f"{val_status} **Content Validation**: {'Enabled' if VALIDATOR_AVAILABLE else 'Disabled'}")
            
            # Tools status
            tools_count = stats.get("tools_available", 0)
            st.sidebar.write(f"🛠️ **Tools Available**: {tools_count}")
            
            # Show multimodal capabilities
            if MULTIMODAL_AVAILABLE:
                with st.sidebar.expander("🎭 Multimodal Capabilities", expanded=False):
                    reader = MultimodalReader()
                    capabilities = reader.check_capabilities()
                    
                    st.write(f"**Vision Model**: {'✅' if capabilities['vision_model_enabled'] else '❌'}")
                    st.write(f"**Video Frames**: {capabilities['video_frame_extraction']}")
                    
                    if capabilities['missing_dependencies']:
                        st.write("**Missing:**")
                        for dep in capabilities['missing_dependencies']:
                            st.write(f"  - {dep}")
                    else:
                        st.write("**Status**: All dependencies available")
            
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
        
        # Neo4j setup
        setup_text = setup_neo4j_docker_instructions()
        st.markdown(setup_text)
        
        # Multimodal setup
        if not MULTIMODAL_AVAILABLE:
            missing_deps = get_missing_dependencies()
            if missing_deps:
                st.markdown("#### 🎭 Multimodal Dependencies")
                st.markdown("Install multimodal support:")
                st.code(f"pip install {' '.join(missing_deps)}")
                
                st.markdown("**Additional system requirements:**")
                st.markdown("- **Tesseract OCR**: For image text extraction")
                st.markdown("  - Ubuntu: `sudo apt install tesseract-ocr`")
                st.markdown("  - macOS: `brew install tesseract`")
                st.markdown("  - Windows: Download from GitHub releases")
                st.markdown("- **FFmpeg**: For video processing (optional)")
                st.markdown("  - Ubuntu: `sudo apt install ffmpeg`")
                st.markdown("  - macOS: `brew install ffmpeg`")
        
        if st.button("Test Connections"):
            with st.spinner("Testing connections..."):
                test_results = test_graph_rag_setup()
                
                st.write("**Test Results:**")
                for key, value in test_results.items():
                    status = "✅" if value else "❌"
                    st.write(f"{status} {key.replace('_', ' ').title()}: {value}")
                
                # Test multimodal
                if MULTIMODAL_AVAILABLE:
                    reader = MultimodalReader()
                    mm_caps = reader.check_capabilities()
                    mm_status = "✅" if mm_caps['multimodal_available'] else "❌"
                    st.write(f"{mm_status} Multimodal Support: {mm_caps['multimodal_available']}")
        
        if st.button("Close Instructions"):
            st.session_state["show_setup"] = False
            st.rerun()


def handle_multimodal_upload(uploaded_file, file_type):
    """Handle multimodal file upload and processing"""
    if not MULTIMODAL_AVAILABLE:
        st.sidebar.error("Multimodal support not available. Install dependencies.")
        return None
    
    try:
        alert = st.sidebar.info(f"Processing {file_type} file...", icon="🎭")
        
        # Get appropriate reader
        reader = get_reader(file_type, uploaded_file.name)
        
        if reader is None:
            st.sidebar.error(f"No reader available for {file_type} files")
            return None
        
        # Process the file
        with st.spinner(f"🎭 Analyzing {file_type} content..."):
            docs = reader.read(uploaded_file)
        
        alert.empty()
        
        if docs and len(docs) > 0:
            # Show processing results in sidebar
            doc = docs[0]
            
            with st.sidebar.expander(f"📄 {uploaded_file.name} Analysis", expanded=True):
                if doc.meta_data:
                    modality = doc.meta_data.get('modality', file_type)
                    st.write(f"**Type**: {modality.title()}")
                    
                    if modality == 'image':
                        if doc.meta_data.get('vision_description'):
                            st.write("**AI Description**: ✅")
                        if doc.meta_data.get('ocr_text'):
                            st.write("**OCR Text**: ✅")
                    
                    elif modality == 'audio':
                        if doc.meta_data.get('transcription'):
                            duration = doc.meta_data.get('duration', 0)
                            st.write(f"**Transcription**: ✅ ({duration:.1f}s)")
                    
                    elif modality == 'video':
                        frames = doc.meta_data.get('frames_count', 0)
                        has_audio = doc.meta_data.get('has_audio', False)
                        st.write(f"**Frames Analyzed**: {frames}")
                        st.write(f"**Audio**: {'✅' if has_audio else '❌'}")
                
                # Show content preview
                content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                st.text_area("Content Preview", content_preview, height=100, disabled=True)
        
        return docs
        
    except Exception as e:
        logger.error(f"Error processing {file_type} file: {e}")
        st.sidebar.error(f"Error processing {file_type}: {str(e)}")
        return None


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>🎭 Multimodal Graph RAG Enterprise</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Advanced RAG with Images, Audio, Video + Neo4j Knowledge Graphs powered by Agno</p>",
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

    # Multimodal toggle
    enable_multimodal = st.sidebar.checkbox(
        "🎭 Enable Multimodal",
        value=MULTIMODAL_AVAILABLE,
        disabled=not MULTIMODAL_AVAILABLE,
        help="Enable image, audio, and video processing"
    )
    
    if not MULTIMODAL_AVAILABLE:
        if st.sidebar.button("🎭 Install Multimodal"):
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
        logger.info(f"---*--- Creating new Multimodal Graph RAG Agent ---*---")
        
        try:
            with st.spinner("🚀 Initializing Multimodal Graph RAG Agent..."):
                agentic_rag_agent = get_agentic_rag_agent(
                    model_id=model_id, 
                    enable_graph=enable_graph,
                    enable_multimodal=enable_multimodal
                )
                
                # Add knowledge validator to prevent hallucination
                if VALIDATOR_AVAILABLE:
                    agentic_rag_agent = add_knowledge_validator_to_agent(agentic_rag_agent)
                    logger.info("✅ Knowledge validator added to agent")
                
                st.session_state["agentic_rag_agent"] = agentic_rag_agent
                st.session_state["current_model"] = model_id
                st.session_state["current_graph_setting"] = enable_graph

                # MANUAL VALIDATOR FIX
                try:
                    from knowledge_validator import KnowledgeBaseValidator
                    
                    # Create validator manually
                    validator_tool = KnowledgeBaseValidator(agentic_rag_agent)
                    
                    # Add to agent tools directly
                    tool_names = [getattr(tool, 'name', str(tool)) for tool in agentic_rag_agent.tools]
                    if 'validate_knowledge_base_contents' not in tool_names:
                        agentic_rag_agent.tools.append(validator_tool)
                        logger.info("✅ Validator tool added manually")
                    
                    # Update instructions
                    validation_instruction = "0. ALWAYS use validate_knowledge_base_contents tool BEFORE making claims about document content"
                    if validation_instruction not in agentic_rag_agent.instructions:
                        agentic_rag_agent.instructions.insert(0, validation_instruction)
                    
                    # Update session state with modified agent
                    st.session_state["agentic_rag_agent"] = agentic_rag_agent
                    
                except Exception as e:
                    logger.error(f"Manual validator addition failed: {e}")
                
                # Success messages
                success_parts = ["📚 Vector RAG"]
                if enable_graph and hasattr(agentic_rag_agent, '_neo4j_client') and agentic_rag_agent._neo4j_client:
                    success_parts.append("🕸️ Knowledge Graph")
                if enable_multimodal:
                    success_parts.append("🎭 Multimodal")
                if VALIDATOR_AVAILABLE:
                    success_parts.append("🛡️ Content Validation")
                
                st.success(f"✅ Agent initialized with {', '.join(success_parts)}!")
                    
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            if "not available" in str(e).lower():
                st.info("💡 Install missing dependencies or check setup instructions.")
                st.session_state["show_setup"] = True
            return
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    ####################################################################
    # Display system status and knowledge base status
    ####################################################################
    display_system_status()
    display_knowledge_base_status()

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

    if prompt := st.chat_input("💬 Ask me anything about your documents, images, audio, or explore the knowledge graph!"):
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

    # File upload with multimodal support
    supported_types = [".pdf", ".csv", ".txt"]
    if enable_multimodal and MULTIMODAL_AVAILABLE:
        supported_types.extend([".jpg", ".jpeg", ".png", ".gif", ".mp3", ".wav", ".mp4", ".mov", ".avi"])
    
    supported_types_str = ", ".join(supported_types)
    
    uploaded_file = st.sidebar.file_uploader(
        f"Add a Document ({supported_types_str})", 
        key="file_upload",
        type=[ext[1:] for ext in supported_types]  # Remove dots for streamlit
    )
    
    if uploaded_file and not prompt:
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in st.session_state.loaded_files:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            # Check if it's a multimodal file
            multimodal_types = ['jpg', 'jpeg', 'png', 'gif', 'mp3', 'wav', 'mp4', 'mov', 'avi']
            
            if file_type in multimodal_types and enable_multimodal:
                docs = handle_multimodal_upload(uploaded_file, file_type)
            else:
                # Standard document processing
                alert = st.sidebar.info("Processing document...", icon="ℹ️")
                reader = get_reader(file_type)
                if reader:
                    docs = reader.read(uploaded_file)
                else:
                    docs = None
                    st.sidebar.error(f"No reader available for .{file_type} files")
                alert.empty()
            
            if docs:
                with st.spinner("📊 Processing and building knowledge graph..."):
                    agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_files.add(file_identifier)
                
                # Show success message based on file type
                if file_type in multimodal_types:
                    st.sidebar.success(f"🎭 {uploaded_file.name} processed and added!")
                else:
                    st.sidebar.success(f"📄 {uploaded_file.name} added to knowledge base")
                
                if enable_graph:
                    st.sidebar.success("🕸️ Knowledge graph updated!")
        else:
            st.sidebar.info(f"{uploaded_file.name} already loaded")

    # Display loaded documents
    if st.session_state.loaded_files or st.session_state.loaded_urls:
        st.sidebar.markdown("#### 📄 Loaded Documents")
        
        if st.session_state.loaded_files:
            st.sidebar.markdown("**Files:**")
            for file_id in st.session_state.loaded_files:
                file_name = file_id.split("_")[0]
                file_ext = file_name.split(".")[-1].lower()
                
                # Choose icon based on file type
                if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                    icon = "🖼️"
                elif file_ext in ['mp3', 'wav', 'm4a', 'flac']:
                    icon = "🎵"
                elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                    icon = "🎬"
                else:
                    icon = "📄"
                
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"{icon} {file_name}")
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
    # Sample Questions - Enhanced with validation warnings
    ###############################################################
    st.sidebar.markdown("#### ❓ Sample Questions")
    
    # Check if knowledge base has content
    has_content = len(st.session_state.loaded_files) > 0 or len(st.session_state.loaded_urls) > 0
    
    if not has_content:
        st.sidebar.warning("⚠️ Upload documents first before asking questions")
    
    # Base questions
    sample_questions = [
        ("📝 Check Content", "What documents are actually in the knowledge base? Use the validation tool first."),
        ("📊 Summarize", "Summarize only the documents that actually exist in the knowledge base"),
        ("🕸️ Entity Map", "What entities are mentioned in the actual uploaded documents?"),
        ("🔍 Find Relations", "Explore relationships from documents that actually exist"),
    ]
    
    # Add multimodal-specific questions if enabled and content exists
    if enable_multimodal and MULTIMODAL_AVAILABLE and has_content:
        # Check if we have multimodal content
        has_multimodal = any(
            file_id.split("_")[0].split(".")[-1].lower() in 
            ['jpg', 'jpeg', 'png', 'gif', 'mp3', 'wav', 'mp4', 'mov', 'avi']
            for file_id in st.session_state.loaded_files
        )
        
        if has_multimodal:
            sample_questions.extend([
                ("🖼️ Image Analysis", "What information was extracted from the actually uploaded images?"),
                ("🎵 Audio Content", "Summarize the transcribed audio content from uploaded files"),
                ("🎬 Video Insights", "What was learned from uploaded video files?"),
                ("🎭 Cross-Modal", "Find connections between uploaded text, image, and audio content"),
            ])
    
    sample_questions.append(("📊 Graph Stats", "Show me knowledge graph statistics"))
    
    for title, question in sample_questions:
        if st.sidebar.button(title, key=f"sample_{hash(question)}", disabled=not has_content and title != "📝 Check Content"):
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
            file_name="multimodal_graph_rag_chat_history.md",
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
            
            # Dynamic spinner text based on enabled features
            spinner_parts = ["🤔 Thinking"]
            if enable_graph:
                spinner_parts.append("🕸️ exploring knowledge graph")
            if enable_multimodal and any(keyword in question.lower() for keyword in ['image', 'audio', 'video', 'visual', 'sound']):
                spinner_parts.append("🎭 analyzing multimodal content")
            if VALIDATOR_AVAILABLE:
                spinner_parts.append("🛡️ validating content")
            
            spinner_text = ", ".join(spinner_parts) + "..."
            
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
    # About section - Enhanced
    ####################################################################
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This **Multimodal Graph RAG Assistant** analyzes documents, images, audio, and video using natural language queries with knowledge graph capabilities.

    **Features:**
    - 📚 Text documents (PDF, TXT, CSV)
    - 🖼️ Image analysis (OCR + AI vision)  
    - 🎵 Audio transcription
    - 🎬 Video frame analysis
    - 🕸️ Neo4j knowledge graphs
    - 🔍 Vector similarity search
    - 🛡️ Content validation (prevents hallucination)

    Built with:
    - 🚀 Agno
    - 💫 Streamlit
    - 🧠 OpenAI GPT-4 Vision & Whisper
    - 🗄️ Qdrant Vector DB
    - 🕸️ Neo4j Graph DB
    """)
    
    # Show current capabilities
    st.sidebar.markdown("**Current Status:**")
    st.sidebar.markdown(f"- Vector Search: {'✅' if True else '❌'}")
    st.sidebar.markdown(f"- Knowledge Graph: {'✅' if enable_graph else '❌'}")
    st.sidebar.markdown(f"- Multimodal: {'✅' if enable_multimodal and MULTIMODAL_AVAILABLE else '❌'}")
    st.sidebar.markdown(f"- Content Validation: {'✅' if VALIDATOR_AVAILABLE else '❌'}")


if __name__ == "__main__":
    main()