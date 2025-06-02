"""
Neo4j Graph RAG Integration Module
Provides Neo4j client and custom Agno tools for knowledge graph operations
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Neo4j imports with error handling
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    ServiceUnavailable = Exception
    AuthError = Exception

# Agno imports
try:
    from agno.tools import Tool
except ImportError:
    # Try alternative import path
    try:
        from agno.tools.base import Tool
    except ImportError:
        # Create a dummy Tool class if not available
        class Tool:
            def __init__(self, name: str, description: str):
                self.name = name
                self.description = description
            
            def run(self, *args, **kwargs):
                return "Tool not available"

from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    """Represents an entity in the knowledge graph"""
    name: str
    entity_type: str
    description: str
    properties: Dict[str, Any]
    source_document: str
    confidence: float = 0.8


@dataclass
class GraphRelationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float = 0.8


class Neo4jClient:
    """Neo4j database client for graph operations"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        """Initialize Neo4j client"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j not available. Install with: pip install neo4j")
            self.driver = None
            return
            
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        if not NEO4J_AVAILABLE:
            return
            
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"❌ Failed to connect to Neo4j: {str(e)}")
            self.driver = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        return self.driver is not None
    
    def create_entity(self, entity: GraphEntity) -> bool:
        """Create or update an entity in the graph"""
        if not self.is_connected():
            return False
        
        try:
            with self.driver.session() as session:
                query = """
                MERGE (e:Entity {name: $name})
                SET e.type = $entity_type,
                    e.description = $description,
                    e.source_document = $source_document,
                    e.confidence = $confidence,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                """
                
                # Add custom properties
                for prop_name, prop_value in entity.properties.items():
                    if isinstance(prop_value, (str, int, float, bool)):
                        query += f", e.{prop_name} = ${prop_name}"
                
                query += " RETURN e"
                
                params = {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "source_document": entity.source_document,
                    "confidence": entity.confidence,
                }
                
                # Add safe properties
                for prop_name, prop_value in entity.properties.items():
                    if isinstance(prop_value, (str, int, float, bool)):
                        params[prop_name] = prop_value
                
                result = session.run(query, params)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error creating entity {entity.name}: {str(e)}")
            return False
    
    def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create relationship between entities"""
        if not self.is_connected():
            return False
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                SET r.confidence = $confidence,
                    r.created_at = datetime()
                RETURN r
                """
                
                params = {
                    "source": relationship.source,
                    "target": relationship.target,
                    "rel_type": relationship.relationship_type,
                    "confidence": relationship.confidence,
                }
                
                result = session.run(query, params)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error creating relationship {relationship.source} -> {relationship.target}: {str(e)}")
            return False
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities in the graph"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query) 
                   OR toLower(e.description) CONTAINS toLower($query)
                   OR toLower(e.type) CONTAINS toLower($query)
                RETURN e
                ORDER BY e.confidence DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {"query": query, "limit": limit})
                
                entities = []
                for record in result:
                    entity = record["e"]
                    entities.append(dict(entity))
                
                return entities
                
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return []
    
    def get_entity_relationships(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get entity and its relationships"""
        if not self.is_connected():
            return {"nodes": [], "relationships": []}
        
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH path = (start:Entity {{name: $entity_name}})-[*1..{depth}]-(connected)
                RETURN path
                LIMIT 50
                """
                
                result = session.run(query, {"entity_name": entity_name})
                
                nodes = {}
                relationships = []
                
                for record in result:
                    path = record["path"]
                    
                    # Collect nodes
                    for node in path.nodes:
                        node_id = node["name"]
                        if node_id not in nodes:
                            nodes[node_id] = dict(node)
                    
                    # Collect relationships
                    for rel in path.relationships:
                        relationships.append({
                            "source": rel.start_node["name"],
                            "target": rel.end_node["name"],
                            "type": rel.get("type", "RELATES"),
                            "properties": dict(rel)
                        })
                
                return {
                    "nodes": list(nodes.values()),
                    "relationships": relationships,
                    "center_entity": entity_name
                }
                
        except Exception as e:
            logger.error(f"Error getting entity relationships: {str(e)}")
            return {"nodes": [], "relationships": []}
    
    def run_cypher_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics"""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            with self.driver.session() as session:
                # Count entities
                entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                
                # Count relationships
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                # Get entity types
                types_result = session.run("""
                    MATCH (e:Entity) 
                    RETURN e.type as type, count(*) as count 
                    ORDER BY count DESC
                """)
                entity_types = [dict(record) for record in types_result]
                
                return {
                    "total_entities": entity_count,
                    "total_relationships": rel_count,
                    "entity_types": entity_types,
                    "connected": True
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {"connected": False, "error": str(e)}


class EntityExtractor:
    """Extracts entities and relationships from text using LLM"""
    
    def __init__(self, model_id: str = "gpt-4o"):
        self.llm = OpenAIChat(id=model_id)
    
    def extract_entities_and_relationships(self, text: str, source_document: str) -> tuple[List[GraphEntity], List[GraphRelationship]]:
        """Extract entities and relationships from text"""
        try:
            prompt = f"""
            Extract entities and relationships from the following text. Return a JSON response with this exact structure:

            {{
                "entities": [
                    {{
                        "name": "entity_name",
                        "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|TECHNOLOGY|EVENT|PRODUCT",
                        "description": "brief description",
                        "properties": {{"key": "value"}}
                    }}
                ],
                "relationships": [
                    {{
                        "source": "entity1_name",
                        "target": "entity2_name", 
                        "type": "WORKS_AT|LOCATED_IN|DEVELOPS|USES|RELATED_TO|PART_OF|MENTIONS",
                        "description": "relationship description",
                        "properties": {{"key": "value"}}
                    }}
                ]
            }}

            Text to analyze:
            {text[:2000]}...

            Focus on extracting meaningful entities and their relationships. Use clear, consistent entity names.
            """
            
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            
            if not response or not response.content:
                return [], []
            
            # Parse JSON response
            try:
                data = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    logger.warning("Could not parse entity extraction response")
                    return [], []
            
            # Convert to GraphEntity objects
            entities = []
            for entity_data in data.get("entities", []):
                entity = GraphEntity(
                    name=entity_data.get("name", "").strip(),
                    entity_type=entity_data.get("type", "CONCEPT"),
                    description=entity_data.get("description", ""),
                    properties=entity_data.get("properties", {}),
                    source_document=source_document,
                    confidence=0.8
                )
                if entity.name:  # Only add entities with names
                    entities.append(entity)
            
            # Convert to GraphRelationship objects
            relationships = []
            for rel_data in data.get("relationships", []):
                relationship = GraphRelationship(
                    source=rel_data.get("source", "").strip(),
                    target=rel_data.get("target", "").strip(),
                    relationship_type=rel_data.get("type", "RELATED_TO"),
                    properties={
                        "description": rel_data.get("description", ""),
                        **rel_data.get("properties", {})
                    },
                    confidence=0.8
                )
                if relationship.source and relationship.target:  # Only add relationships with source and target
                    relationships.append(relationship)
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return [], []


class Neo4jSearchTool(Tool):
    """Custom Agno tool for searching the Neo4j knowledge graph"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__(
            name="search_knowledge_graph",
            description="Search the knowledge graph for entities, relationships, and concepts. Use this to find connections between different pieces of information.",
        )
        self.neo4j_client = neo4j_client
    
    def run(self, query: str) -> str:
        """Search the knowledge graph"""
        try:
            if not self.neo4j_client.is_connected():
                return "Knowledge graph is not available. Make sure Neo4j is running."
            
            # Search for entities
            entities = self.neo4j_client.search_entities(query, limit=5)
            
            if not entities:
                return f"No entities found in the knowledge graph for query: '{query}'"
            
            # Format results
            result = f"Knowledge Graph Search Results for '{query}':\n\n"
            
            for i, entity in enumerate(entities, 1):
                result += f"{i}. **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Unknown')})\n"
                result += f"   Description: {entity.get('description', 'No description')}\n"
                result += f"   Source: {entity.get('source_document', 'Unknown')}\n"
                result += f"   Confidence: {entity.get('confidence', 0.0):.2f}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error searching knowledge graph: {str(e)}"


class Neo4jRelationshipTool(Tool):
    """Custom Agno tool for exploring entity relationships in Neo4j"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__(
            name="explore_entity_relationships", 
            description="Explore relationships and connections for a specific entity in the knowledge graph. Useful for understanding how concepts are connected.",
        )
        self.neo4j_client = neo4j_client
    
    def run(self, entity_name: str) -> str:
        """Explore entity relationships"""
        try:
            if not self.neo4j_client.is_connected():
                return "Knowledge graph is not available. Make sure Neo4j is running."
            
            # Get entity relationships
            graph_data = self.neo4j_client.get_entity_relationships(entity_name, depth=2)
            
            if not graph_data["nodes"]:
                return f"No entity found with name '{entity_name}' in the knowledge graph."
            
            # Format results
            result = f"Entity Relationship Map for '{entity_name}':\n\n"
            
            # Show connected entities
            result += "**Connected Entities:**\n"
            for node in graph_data["nodes"]:
                if node["name"] != entity_name:
                    result += f"- {node['name']} ({node.get('type', 'Unknown')})\n"
            
            result += f"\n**Relationships:**\n"
            for rel in graph_data["relationships"]:
                result += f"- {rel['source']} → {rel['target']} ({rel['type']})\n"
            
            return result
            
        except Exception as e:
            return f"Error exploring entity relationships: {str(e)}"


class Neo4jStatsTool(Tool):
    """Custom Agno tool for getting graph statistics"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__(
            name="get_graph_statistics",
            description="Get statistics and overview of the knowledge graph including entity counts, types, and relationships.",
        )
        self.neo4j_client = neo4j_client
    
    def run(self) -> str:
        """Get graph statistics"""
        try:
            stats = self.neo4j_client.get_graph_statistics()
            
            if not stats.get("connected", False):
                return f"Knowledge graph is not connected. Error: {stats.get('error', 'Neo4j not available')}"
            
            result = "📊 **Knowledge Graph Statistics:**\n\n"
            result += f"Total Entities: {stats['total_entities']}\n"
            result += f"Total Relationships: {stats['total_relationships']}\n\n"
            
            result += "**Entity Types:**\n"
            for entity_type in stats.get('entity_types', []):
                result += f"- {entity_type.get('type', 'Unknown')}: {entity_type.get('count', 0)} entities\n"
            
            return result
            
        except Exception as e:
            return f"Error getting graph statistics: {str(e)}"


def create_neo4j_tools(neo4j_uri: str = "bolt://localhost:7687", 
                      neo4j_user: str = "neo4j", 
                      neo4j_password: str = "password") -> tuple[List[Tool], Optional[Neo4jClient]]:
    """Create Neo4j tools for Agno agent"""
    
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j not available - skipping graph tools")
        return [], None
    
    # Initialize Neo4j client
    neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    
    if not neo4j_client.is_connected():
        logger.warning("Neo4j not connected - skipping graph tools")
        return [], None
    
    # Create tools
    tools = [
        Neo4jSearchTool(neo4j_client),
        Neo4jRelationshipTool(neo4j_client),
        Neo4jStatsTool(neo4j_client)
    ]
    
    return tools, neo4j_client


def test_neo4j_connection(uri: str = "bolt://localhost:7687", 
                         user: str = "neo4j", 
                         password: str = "password") -> Dict[str, Any]:
    """Test Neo4j connection and return status"""
    
    if not NEO4J_AVAILABLE:
        return {
            "connected": False,
            "message": "Neo4j library not installed. Install with: pip install neo4j",
            "neo4j_available": False
        }
    
    client = Neo4jClient(uri, user, password)
    
    if client.is_connected():
        stats = client.get_graph_statistics()
        client.close()
        return {
            "connected": True,
            "message": "Successfully connected to Neo4j",
            "statistics": stats,
            "neo4j_available": True
        }
    else:
        return {
            "connected": False,
            "message": "Failed to connect to Neo4j. Make sure Neo4j is running and credentials are correct.",
            "neo4j_available": True,
            "help": """
To set up Neo4j locally:
1. Install Neo4j Desktop or use Docker:
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
2. Access Neo4j Browser at http://localhost:7474
3. Update password if needed
4. Verify connection settings in your application
            """
        }


if __name__ == "__main__":
    # Test the Neo4j integration
    print("Testing Neo4j Graph RAG Integration...")
    
    # Test connection
    connection_test = test_neo4j_connection()
    print(f"Connection test: {connection_test}")
    
    if connection_test["connected"]:
        print("✅ Neo4j integration ready!")
        print("You can now use the Graph RAG agent with Neo4j tools.")
    else:
        print("❌ Neo4j setup required.")
        print(connection_test.get("help", ""))