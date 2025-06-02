"""🤖 Graph RAG Agent with Neo4j Integration and Multimodal Support"""

import os
from typing import Optional, List
import logging

from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge import AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.qdrant import Qdrant
from agno.document import Document

# Import models with error handling
try:
    from agno.models.anthropic import Claude
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from agno.models.google import Gemini
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from agno.models.groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Import Neo4j components
try:
    from neo4j_integration import (
        create_neo4j_tools,
        test_neo4j_connection,
        EntityExtractor,
        Neo4jClient,
        NEO4J_AVAILABLE
    )
    GRAPH_RAG_AVAILABLE = True
except ImportError:
    GRAPH_RAG_AVAILABLE = False
    NEO4J_AVAILABLE = False

# Import Multimodal components
try:
    from multimodal_processors import (
        ImageProcessor, AudioProcessor, VideoProcessor,
        MultimodalDocumentLoader, get_missing_dependencies
    )
    from multimodal_readers import (
        MultimodalReader, ImageReader, AudioReader, VideoReader
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# Configuration
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_agentic_rag_agent(
    model_id: str = "openai:gpt-4o",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
    enable_graph: bool = True,
    enable_multimodal: bool = True,
) -> Agent:
    """Get Graph RAG Agent with optional multimodal support"""
    
    provider, model_name = model_id.split(":")

    if provider == "openai":
        model = OpenAIChat(id=model_name)
    elif provider == "google" and GOOGLE_AVAILABLE:
        model = Gemini(id=model_name)
    elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
        model = Claude(id=model_name)
    elif provider == "groq" and GROQ_AVAILABLE:
        model = Groq(id=model_name)
    else:
        raise ValueError(f"Model {model_id} not available")

    # Vector database
    vector_db = Qdrant(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection="graph_rag_docs",
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    )

    # Standard knowledge base
    knowledge = AgentKnowledge(
        vector_db=vector_db,
        num_documents=10
    )

    # Neo4j tools
    neo4j_tools = []
    neo4j_client = None
    
    if enable_graph and GRAPH_RAG_AVAILABLE:
        try:
            test_result = test_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
            if test_result["connected"]:
                neo4j_tools, neo4j_client = create_neo4j_tools(neo4j_uri, neo4j_user, neo4j_password)
                logger.info("✅ Graph RAG enabled")
            else:
                logger.warning("❌ Neo4j not connected")
        except Exception as e:
            logger.error(f"Neo4j failed: {e}")

    # Tools
    all_tools = [DuckDuckGoTools()]
    all_tools.extend(neo4j_tools)

    # Enhanced instructions for multimodal support
    base_instructions = [
        "1. ALWAYS search knowledge base thoroughly and report EXACTLY what documents exist", 
        "2. When summarizing, analyze ONLY the documents that actually exist in the knowledge base", 
        "3. Use graph tools for entities only when documents are present", 
        "4. ALWAYS mention the exact document count and types",
        "5. NEVER assume or fabricate information about images, audio, or video unless they were explicitly uploaded",
        "6. If asked about multimodal content but none exists, clearly state 'No images, audio, or video files have been uploaded to the knowledge base'"
    ]
    
    if enable_multimodal and MULTIMODAL_AVAILABLE:
        base_instructions.extend([
            "7. When analyzing images, mention both OCR text and AI descriptions ONLY if images were actually uploaded",
            "8. For audio content, reference transcription quality and duration ONLY if audio files exist",
            "9. For video content, describe both visual frames and audio content ONLY if video files exist", 
            "10. Look for cross-modal connections ONLY between actually uploaded content types",
            "11. Be precise about what file types are actually in the knowledge base before analyzing"
        ])
    else:
        base_instructions.extend([
            "7. Multimodal processing is not enabled - only analyze text documents"
        ])

    # Agent description
    agent_description = "Graph RAG Assistant"
    if enable_multimodal and MULTIMODAL_AVAILABLE:
        agent_description = "Multimodal Graph RAG Assistant with image, audio, and video analysis capabilities"

    # Agent
    agent = Agent(
        name="graph_rag_agent",
        session_id=session_id,
        user_id=user_id,
        model=model,
        storage=PostgresAgentStorage(table_name="graph_rag_sessions", db_url=db_url),
        knowledge=knowledge,
        description=agent_description,
        instructions=base_instructions,
        search_knowledge=True,
        read_chat_history=True,
        tools=all_tools,
        markdown=True,
        show_tool_calls=True,
        add_history_to_messages=True,
        debug_mode=debug_mode,
    )

    # Add custom attributes
    agent._neo4j_client = neo4j_client
    agent._multimodal_enabled = enable_multimodal and MULTIMODAL_AVAILABLE
    
    return agent


def get_graph_rag_statistics(agent):
    """Get statistics including multimodal capabilities"""
    stats = {
        "vector_db_enabled": True,
        "graph_db_enabled": bool(getattr(agent, '_neo4j_client', None)),
        "multimodal_enabled": getattr(agent, '_multimodal_enabled', False),
        "tools_available": len(agent.tools),
        "graph_entities": 0,
        "graph_relationships": 0,
    }
    
    # Get actual graph statistics if Neo4j is connected
    if stats["graph_db_enabled"]:
        try:
            neo4j_client = getattr(agent, '_neo4j_client', None)
            if neo4j_client:
                graph_stats = neo4j_client.get_graph_statistics()
                if graph_stats.get("connected"):
                    stats["graph_entities"] = graph_stats.get("total_entities", 0)
                    stats["graph_relationships"] = graph_stats.get("total_relationships", 0)
        except Exception as e:
            logger.warning(f"Could not get graph statistics: {e}")
    
    return stats


def test_graph_rag_setup():
    """Test all system components including multimodal"""
    return {
        "neo4j_integration": GRAPH_RAG_AVAILABLE,
        "neo4j_library": NEO4J_AVAILABLE,
        "multimodal_available": MULTIMODAL_AVAILABLE,
        "anthropic_available": ANTHROPIC_AVAILABLE,
        "google_available": GOOGLE_AVAILABLE,
        "groq_available": GROQ_AVAILABLE,
    }


def setup_neo4j_docker_instructions():
    """Return setup instructions for Neo4j and multimodal components"""
    instructions = """
🐳 **Neo4j Setup:**
```bash
pip install neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

🎭 **Multimodal Setup:**
```bash
pip install Pillow pytesseract librosa soundfile opencv-python
```

**System Dependencies:**
- **Tesseract OCR**: `brew install tesseract` (macOS) or `sudo apt install tesseract-ocr` (Ubuntu)
- **FFmpeg** (optional): `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu)
"""
    return instructions


def get_multimodal_capabilities():
    """Get detailed multimodal capabilities"""
    if not MULTIMODAL_AVAILABLE:
        return {
            "available": False,
            "missing_dependencies": ["multimodal_processors", "multimodal_readers"]
        }
    
    try:
        from multimodal_readers import MultimodalReader
        reader = MultimodalReader()
        return reader.check_capabilities()
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def create_multimodal_document_from_file(file_path: str) -> Optional[Document]:
    """Create document from multimodal file"""
    if not MULTIMODAL_AVAILABLE:
        logger.warning("Multimodal processing not available")
        return None
    
    try:
        from multimodal_readers import MultimodalReader
        reader = MultimodalReader()
        docs = reader.read(file_path)
        return docs[0] if docs else None
    except Exception as e:
        logger.error(f"Error processing multimodal file {file_path}: {e}")
        return None


if __name__ == "__main__":
    # Test the complete system
    print("Testing Multimodal Graph RAG System...")
    
    # Test system capabilities
    test_results = test_graph_rag_setup()
    print(f"System test results: {test_results}")
    
    # Test multimodal capabilities
    if MULTIMODAL_AVAILABLE:
        mm_caps = get_multimodal_capabilities()
        print(f"Multimodal capabilities: {mm_caps}")
    
    # Try to create an agent
    try:
        agent = get_agentic_rag_agent(
            enable_graph=GRAPH_RAG_AVAILABLE,
            enable_multimodal=MULTIMODAL_AVAILABLE
        )
        stats = get_graph_rag_statistics(agent)
        print(f"Agent statistics: {stats}")
        print("✅ Multimodal Graph RAG system is ready!")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")