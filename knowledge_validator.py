"""
Knowledge Base Content Validator - Fixed for Qdrant API
Prevents hallucination by validating what's actually in the knowledge base
"""

from typing import Dict, List, Any, Optional
from agno.agent import Agent

# Try different import paths for Tool
try:
    from agno.tools.base import Tool
except ImportError:
    try:
        from agno.tool import Tool
    except ImportError:
        try:
            # If Tool is not available, create a simple base class
            class Tool:
                def __init__(self, name: str, description: str):
                    self.name = name
                    self.description = description
                
                def run(self, *args, **kwargs):
                    return "Tool execution not implemented"
        except:
            # Last resort - create a minimal Tool class
            class Tool:
                def __init__(self, name: str, description: str):
                    self.name = name
                    self.description = description
                
                def run(self, *args, **kwargs):
                    return "Tool execution not implemented"


class KnowledgeBaseValidator(Tool):
    """Tool to validate and report actual knowledge base contents"""
    
    def __init__(self, agent: Agent):
        super().__init__(
            name="validate_knowledge_base_contents",
            description="Check what documents are actually stored in the knowledge base and their types. Use this BEFORE making claims about document content."
        )
        self.agent = agent
    
    def run(self) -> str:
        """Validate and return actual knowledge base contents"""
        try:
            if not hasattr(self.agent, 'knowledge') or not self.agent.knowledge:
                return "❌ No knowledge base found."
            
            # Try to get documents from the knowledge base
            if hasattr(self.agent.knowledge, 'vector_db') and self.agent.knowledge.vector_db:
                try:
                    # Query the vector database to see what's actually stored
                    vector_db = self.agent.knowledge.vector_db
                    
                    # Get collection info if available
                    collection_info = self._get_collection_info(vector_db)
                    
                    return f"""📊 **Knowledge Base Validation Report:**

**Collection Status:** {collection_info['status']}
**Total Documents:** {collection_info['document_count']}
**Document Types Found:** {', '.join(collection_info['document_types']) if collection_info['document_types'] else 'None'}

**Document Breakdown:**
{self._format_document_breakdown(collection_info)}

**Available Modalities:**
- Text Documents: {collection_info['text_count']}
- Image Documents: {collection_info['image_count']} 
- Audio Documents: {collection_info['audio_count']}
- Video Documents: {collection_info['video_count']}

**⚠️ Important:** Only analyze content that actually exists in the knowledge base. Do not fabricate information about non-existent files."""
                    
                except Exception as e:
                    return f"❌ Error accessing knowledge base: {str(e)}"
            else:
                return "❌ No vector database found in knowledge base."
                
        except Exception as e:
            return f"❌ Knowledge base validation failed: {str(e)}"
    
    def _get_collection_info(self, vector_db) -> Dict[str, Any]:
        """Get information about what's actually in the vector database - Fixed for Qdrant API"""
        info = {
            'status': 'Unknown',
            'document_count': 0,
            'document_types': [],
            'text_count': 0,
            'image_count': 0,
            'audio_count': 0,
            'video_count': 0,
            'documents': []
        }
        
        try:
            # Try to get collection info from Qdrant
            if hasattr(vector_db, 'client') and vector_db.client:
                try:
                    collection_name = getattr(vector_db, 'collection', 'graph_rag_docs')
                    
                    # First, check if collection exists
                    try:
                        collections = vector_db.client.get_collections()
                        collection_exists = any(c.name == collection_name for c in collections.collections)
                        
                        if not collection_exists:
                            info['status'] = f'Collection "{collection_name}" does not exist'
                            return info
                            
                    except Exception as e:
                        info['status'] = f'Cannot list collections: {str(e)}'
                        return info
                    
                    # Get collection info using the correct API
                    try:
                        collection_info = vector_db.client.get_collection(collection_name)
                        info['status'] = 'Active'
                        
                        # Try different ways to get point count
                        if hasattr(collection_info, 'points_count'):
                            info['document_count'] = collection_info.points_count
                        elif hasattr(collection_info, 'point_count'):
                            info['document_count'] = collection_info.point_count
                        else:
                            # Fallback: count points manually
                            try:
                                count_result = vector_db.client.count(collection_name)
                                if hasattr(count_result, 'count'):
                                    info['document_count'] = count_result.count
                                else:
                                    info['document_count'] = count_result
                            except Exception as count_error:
                                info['status'] = f'Active (count error: {str(count_error)})'
                                info['document_count'] = 0
                        
                        # Try to get some sample points to analyze document types
                        if info['document_count'] > 0:
                            try:
                                # Use scroll to get sample points
                                scroll_result = vector_db.client.scroll(
                                    collection_name=collection_name,
                                    limit=min(10, info['document_count']),
                                    with_payload=True
                                )
                                
                                # Handle different scroll result formats
                                points = []
                                if isinstance(scroll_result, tuple):
                                    points = scroll_result[0]  # (points, next_page_offset)
                                elif hasattr(scroll_result, 'points'):
                                    points = scroll_result.points
                                else:
                                    points = scroll_result
                                
                                for point in points:
                                    if hasattr(point, 'payload') and point.payload:
                                        doc_type = self._determine_document_type(point.payload)
                                        info['documents'].append({
                                            'type': doc_type,
                                            'payload': point.payload
                                        })
                                        
                                        # Count by type
                                        if doc_type == 'text':
                                            info['text_count'] += 1
                                        elif doc_type == 'image':
                                            info['image_count'] += 1
                                        elif doc_type == 'audio':
                                            info['audio_count'] += 1
                                        elif doc_type == 'video':
                                            info['video_count'] += 1
                                        
                                        if doc_type not in info['document_types']:
                                            info['document_types'].append(doc_type)
                                
                            except Exception as e:
                                info['status'] = f'Active (analysis error: {str(e)})'
                                
                    except Exception as e:
                        info['status'] = f'Collection access error: {str(e)}'
                        
                except Exception as e:
                    info['status'] = f'Client error: {str(e)}'
                    
        except Exception as e:
            info['status'] = f'Failed to access: {str(e)}'
        
        return info
    
    def _determine_document_type(self, payload: Dict) -> str:
        """Determine document type from payload metadata - Fixed to check nested meta_data"""
        
        # First check if meta_data exists and contains type information
        if 'meta_data' in payload and isinstance(payload['meta_data'], dict):
            meta_data = payload['meta_data']
            
            # Check for explicit type markers in meta_data
            if 'type' in meta_data:
                doc_type = meta_data['type']
                if isinstance(doc_type, str) and doc_type.lower() in ['image', 'audio', 'video']:
                    return doc_type.lower()
            
            if 'modality' in meta_data:
                modality = meta_data['modality']
                if isinstance(modality, str) and modality.lower() in ['image', 'audio', 'video']:
                    return modality.lower()
            
            # Check for file extensions or source indicators in meta_data
            if 'source_file' in meta_data:
                source = meta_data['source_file']
                if isinstance(source, str):
                    source_lower = source.lower()
                    if any(ext in source_lower for ext in ['.jpg', '.png', '.gif', '.jpeg', '.bmp']):
                        return 'image'
                    elif any(ext in source_lower for ext in ['.mp3', '.wav', '.m4a', '.flac']):
                        return 'audio'
                    elif any(ext in source_lower for ext in ['.mp4', '.avi', '.mov', '.mkv']):
                        return 'video'
            
            # Check for processing method indicators in meta_data
            if 'processing_method' in meta_data:
                method = meta_data['processing_method']
                if isinstance(method, str):
                    method_lower = method.lower()
                    if 'ocr' in method_lower or 'vision' in method_lower:
                        return 'image'
                    elif 'transcription' in method_lower or 'whisper' in method_lower:
                        return 'audio'
                    elif 'frame' in method_lower and 'audio' in method_lower:
                        return 'video'
        
        # Fallback: Check top-level payload (for backwards compatibility)
        # Check for explicit type markers
        if 'type' in payload:
            doc_type = payload['type']
            if isinstance(doc_type, str) and doc_type.lower() in ['image', 'audio', 'video']:
                return doc_type.lower()
        
        if 'modality' in payload:
            modality = payload['modality']
            if isinstance(modality, str) and modality.lower() in ['image', 'audio', 'video']:
                return modality.lower()
        
        # Check for file extensions or source indicators
        if 'source_file' in payload:
            source = payload['source_file']
            if isinstance(source, str):
                source_lower = source.lower()
                if any(ext in source_lower for ext in ['.jpg', '.png', '.gif', '.jpeg', '.bmp']):
                    return 'image'
                elif any(ext in source_lower for ext in ['.mp3', '.wav', '.m4a', '.flac']):
                    return 'audio'
                elif any(ext in source_lower for ext in ['.mp4', '.avi', '.mov', '.mkv']):
                    return 'video'
        
        # Check for processing method indicators
        if 'processing_method' in payload:
            method = payload['processing_method']
            if isinstance(method, str):
                method_lower = method.lower()
                if 'ocr' in method_lower or 'vision' in method_lower:
                    return 'image'
                elif 'transcription' in method_lower or 'whisper' in method_lower:
                    return 'audio'
                elif 'frame' in method_lower and 'audio' in method_lower:
                    return 'video'
        
        # Default to text
        return 'text'
    
    def _format_document_breakdown(self, info: Dict) -> str:
        """Format the document breakdown for display"""
        if not info['documents']:
            return "- No documents found"
        
        breakdown = []
        for i, doc in enumerate(info['documents'][:5], 1):  # Show first 5
            doc_type = doc['type']
            payload = doc['payload']
            
            # Get source file name if available
            source = payload.get('source_file', payload.get('file_path', 'Unknown'))
            
            breakdown.append(f"- Document {i}: {doc_type.title()} ({source})")
        
        if len(info['documents']) > 5:
            breakdown.append(f"- ... and {len(info['documents']) - 5} more documents")
        
        return '\n'.join(breakdown)


def add_knowledge_validator_to_agent(agent: Agent) -> Agent:
    """Add the knowledge base validator tool to an existing agent"""
    
    validator = KnowledgeBaseValidator(agent)
    
    # Add to tools if not already present
    if hasattr(agent, 'tools') and agent.tools:
        tool_names = [getattr(tool, 'name', str(tool)) for tool in agent.tools]
        if validator.name not in tool_names:
            agent.tools.append(validator)
            
            # Update instructions to use the validator
            current_instructions = agent.instructions or []
            
            # Add validation instruction at the beginning
            validation_instruction = "0. ALWAYS use validate_knowledge_base_contents tool BEFORE making claims about document content"
            
            if validation_instruction not in current_instructions:
                agent.instructions = [validation_instruction] + current_instructions
    else:
        # If no tools exist, create the tools list
        agent.tools = [validator]
        
        # Add instructions
        validation_instruction = "0. ALWAYS use validate_knowledge_base_contents tool BEFORE making claims about document content"
        current_instructions = agent.instructions or []
        agent.instructions = [validation_instruction] + current_instructions
    
    return agent


# Test function to check if the validator works
def test_validator():
    """Test the validator independently"""
    print("🧪 Testing Fixed Knowledge Validator")
    
    try:
        # Test with a real agent if possible
        from agentic_rag import get_agentic_rag_agent
        
        agent = get_agentic_rag_agent()
        validator = KnowledgeBaseValidator(agent)
        
        print(f"✅ Validator created: {validator.name}")
        
        # Test the run method
        result = validator.run()
        print(f"✅ Validator run result:")
        print(result)
        
        return True
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the validator
    print("Knowledge Base Validator - Fixed for Qdrant API")
    print("This tool helps prevent AI hallucination about non-existent documents")
    print("-" * 60)
    
    success = test_validator()
    if success:
        print("🎉 Fixed validator is working!")
    else:
        print("❌ Validator still has issues - check the errors above")