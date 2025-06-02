"""🤖 Graph RAG Agent with Neo4j Integration"""

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
) -> Agent:
    """Get Graph RAG Agent"""
    
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

    # Standard knowledge base - NO CUSTOM CLASS
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

    # Agent
    agent = Agent(
        name="graph_rag_agent",
        session_id=session_id,
        user_id=user_id,
        model=model,
        storage=PostgresAgentStorage(table_name="graph_rag_sessions", db_url=db_url),
        knowledge=knowledge,
        description="Graph RAG Assistant",
        instructions=["1. ALWAYS search knowledge base thoroughly", "2. When summarizing, analyze ALL documents", 
                      "3. Use graph tools for entities", "4. Mention document count"],
        search_knowledge=True,
        read_chat_history=True,
        tools=all_tools,
        markdown=True,
        show_tool_calls=True,
        add_history_to_messages=True,
        debug_mode=debug_mode,
    )

    agent._neo4j_client = neo4j_client
    return agent


def get_graph_rag_statistics(agent):
    """Get statistics"""
    return {
        "vector_db_enabled": True,
        "graph_db_enabled": bool(getattr(agent, '_neo4j_client', None)),
        "tools_available": len(agent.tools),
        "graph_entities": 0,
        "graph_relationships": 0,
    }


def test_graph_rag_setup():
    return {
        "neo4j_integration": GRAPH_RAG_AVAILABLE,
        "neo4j_library": NEO4J_AVAILABLE,
        "anthropic_available": ANTHROPIC_AVAILABLE,
        "google_available": GOOGLE_AVAILABLE,
        "groq_available": GROQ_AVAILABLE,
    }


def setup_neo4j_docker_instructions():
    return """
🐳 **Neo4j Setup:**
```bash
pip install neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
"""