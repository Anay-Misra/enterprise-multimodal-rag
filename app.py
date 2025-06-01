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
from agentic_rag import get_agentic_rag_agent
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

# Import evaluation module
try:
    from rag_evaluator import RAGEvaluator, RAGEvaluationDataset
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    st.warning("⚠️ Evaluation module not found. Please create rag_evaluator.py for evaluation features.")

nest_asyncio.apply()
st.set_page_config(
    page_title="EnterpriseGPT",
    page_icon="💎",
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


def run_evaluation_interface():
    """Create evaluation interface in sidebar"""
    if not EVALUATION_AVAILABLE:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Evaluation & Testing")
    
    # Initialize evaluation state
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []
    
    # Evaluation options
    eval_type = st.sidebar.selectbox(
        "Evaluation Type",
        ["Quick Evaluation", "Comprehensive", "Model Comparison", "Custom Dataset"],
        key="eval_type_selector"
    )
    
    # Quick Evaluation
    if eval_type == "Quick Evaluation":
        if st.sidebar.button("🚀 Run Quick Eval", key="quick_eval_btn"):
            run_quick_evaluation()
    
    # Comprehensive Evaluation  
    elif eval_type == "Comprehensive":
        num_questions = st.sidebar.slider("Number of Questions", 3, 10, 5)
        if st.sidebar.button("📊 Run Comprehensive", key="comp_eval_btn"):
            run_comprehensive_evaluation(num_questions)
    
    # Model Comparison
    elif eval_type == "Model Comparison":
        available_models = [
            "openai:gpt-4o",
            "openai:o3-mini", 
            "anthropic:claude-3-5-sonnet-20241022",
            "google:gemini-2.0-flash-exp",
            "groq:llama-3.3-70b-versatile"
        ]
        
        selected_models = st.sidebar.multiselect(
            "Select Models to Compare",
            available_models,
            default=available_models[:2]
        )
        
        if st.sidebar.button("🔄 Compare Models", key="compare_models_btn") and selected_models:
            run_model_comparison(selected_models)
    
    # Custom Dataset
    elif eval_type == "Custom Dataset":
        uploaded_eval_file = st.sidebar.file_uploader(
            "Upload Evaluation Dataset (JSON)",
            type=["json"],
            key="eval_dataset_upload"
        )
        
        if uploaded_eval_file and st.sidebar.button("📝 Run Custom Eval", key="custom_eval_btn"):
            run_custom_evaluation(uploaded_eval_file)
    
    # Display recent results
    if st.session_state.evaluation_results:
        st.sidebar.markdown("#### 📈 Recent Results")
        latest_result = st.session_state.evaluation_results[-1]
        
        with st.sidebar.expander("Latest Evaluation", expanded=False):
            st.write(f"**Timestamp:** {latest_result.get('timestamp', 'Unknown')}")
            if latest_result.get('accuracy_score'):
                st.write(f"**Accuracy:** {latest_result['accuracy_score']:.2f}/10")
            if latest_result.get('reliability_score'):
                st.write(f"**Reliability:** {latest_result['reliability_score']:.1f}%")
        
        # Download results
        if st.sidebar.download_button(
            "💾 Download Results",
            json.dumps(st.session_state.evaluation_results, indent=2),
            file_name=f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_eval_results"
        ):
            st.sidebar.success("📊 Evaluation results downloaded!")


def run_quick_evaluation():
    """Run a quick 3-question evaluation"""
    try:
        with st.spinner("🔍 Running Quick Evaluation..."):
            agent = st.session_state.get("agentic_rag_agent")
            if not agent:
                st.error("❌ No agent found. Please initialize the system first.")
                return
            
            evaluator = RAGEvaluator(agent)
            
            # Create mini dataset for quick eval that requires knowledge base usage
            quick_dataset = RAGEvaluationDataset(
                questions=[
                    "What specific content is in the knowledge base? Use search_knowledge_base tool.",
                    "Can you summarize the main topics from the documents? Search the knowledge base.",
                    "What can you tell me about the content you have access to? Use the search tool."
                ],
                contexts=[
                    "Knowledge base contains uploaded documents and web content",
                    "System uses RAG with vector search and language models",
                    "Assistant helps with document analysis and question answering"
                ],
                expected_answers=[
                    "The knowledge base contains specific documents that should be retrieved and described",
                    "Main topics should be extracted from actual knowledge base content",
                    "Content description should be based on actual knowledge base search results"
                ],
                ground_truths=[
                    "Document repository with user uploads",
                    "Knowledge base topic summary",
                    "Available content description"
                ]
            )
            
            result = evaluator.run_comprehensive_evaluation(quick_dataset)
            
            # Store result
            result_dict = {
                "timestamp": result.timestamp,
                "type": "Quick Evaluation",
                "accuracy_score": result.accuracy_score,
                "reliability_score": result.reliability_score,
                "performance_metrics": result.performance_metrics,
                "custom_metrics": result.custom_metrics
            }
            st.session_state.evaluation_results.append(result_dict)
            
            # Display results
            st.success("✅ Quick evaluation completed!")
            display_evaluation_results(result_dict)
            
    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")


def run_comprehensive_evaluation(num_questions: int):
    """Run comprehensive evaluation with specified number of questions"""
    try:
        with st.spinner(f"📊 Running Comprehensive Evaluation ({num_questions} questions)..."):
            agent = st.session_state.get("agentic_rag_agent")
            if not agent:
                st.error("❌ No agent found. Please initialize the system first.")
                return
            
            evaluator = RAGEvaluator(agent)
            result = evaluator.run_comprehensive_evaluation()
            
            # Store result
            result_dict = {
                "timestamp": result.timestamp,
                "type": f"Comprehensive ({num_questions} questions)",
                "accuracy_score": result.accuracy_score,
                "reliability_score": result.reliability_score,
                "performance_metrics": result.performance_metrics,
                "custom_metrics": result.custom_metrics
            }
            st.session_state.evaluation_results.append(result_dict)
            
            st.success("✅ Comprehensive evaluation completed!")
            display_evaluation_results(result_dict)
            
    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")


def run_model_comparison(selected_models: List[str]):
    """Run comparison between multiple models"""
    try:
        with st.spinner(f"🔄 Comparing {len(selected_models)} models..."):
            agent = st.session_state.get("agentic_rag_agent")
            if not agent:
                st.error("❌ No agent found. Please initialize the system first.")
                return
            
            evaluator = RAGEvaluator(agent)
            results = evaluator.compare_models(selected_models)
            
            # Store comparison results
            comparison_dict = {
                "timestamp": datetime.now().isoformat(),
                "type": "Model Comparison",
                "models": selected_models,
                "results": {}
            }
            
            for model_id, result in results.items():
                comparison_dict["results"][model_id] = {
                    "accuracy_score": result.accuracy_score,
                    "reliability_score": result.reliability_score,
                    "performance_metrics": result.performance_metrics
                }
            
            st.session_state.evaluation_results.append(comparison_dict)
            
            st.success("✅ Model comparison completed!")
            display_comparison_results(comparison_dict)
            
    except Exception as e:
        st.error(f"❌ Model comparison failed: {str(e)}")


def run_custom_evaluation(uploaded_file):
    """Run evaluation with custom dataset"""
    try:
        with st.spinner("📝 Running Custom Evaluation..."):
            # Load custom dataset
            dataset_content = json.loads(uploaded_file.read())
            custom_dataset = RAGEvaluationDataset(**dataset_content)
            
            agent = st.session_state.get("agentic_rag_agent")
            if not agent:
                st.error("❌ No agent found. Please initialize the system first.")
                return
            
            evaluator = RAGEvaluator(agent)
            result = evaluator.run_comprehensive_evaluation(custom_dataset)
            
            # Store result
            result_dict = {
                "timestamp": result.timestamp,
                "type": "Custom Dataset",
                "dataset_size": len(custom_dataset.questions),
                "accuracy_score": result.accuracy_score,
                "reliability_score": result.reliability_score,
                "performance_metrics": result.performance_metrics,
                "custom_metrics": result.custom_metrics
            }
            st.session_state.evaluation_results.append(result_dict)
            
            st.success("✅ Custom evaluation completed!")
            display_evaluation_results(result_dict)
            
    except Exception as e:
        st.error(f"❌ Custom evaluation failed: {str(e)}")


def display_evaluation_results(result_dict):
    """Display evaluation results in the main area"""
    st.markdown("### 📊 Evaluation Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result_dict.get('accuracy_score'):
            st.metric(
                "Accuracy Score", 
                f"{result_dict['accuracy_score']:.2f}/10",
                delta=None
            )
    
    with col2:
        if result_dict.get('reliability_score'):
            st.metric(
                "Reliability", 
                f"{result_dict['reliability_score']:.1f}%",
                delta=None
            )
    
    with col3:
        if result_dict.get('performance_metrics'):
            avg_latency = result_dict['performance_metrics'].get('avg_latency_seconds', 0)
            st.metric(
                "Avg Response Time", 
                f"{avg_latency:.2f}s",
                delta=None
            )
    
    # Detailed metrics
    if result_dict.get('custom_metrics'):
        st.markdown("#### 🎨 Custom RAG Metrics")
        custom = result_dict['custom_metrics']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Knowledge Base Usage:** {custom.get('knowledge_base_usage', 0):.1f}%")
            st.write(f"**Response Completeness:** {custom.get('response_completeness', 0):.1f}%")
        
        with col2:
            st.write(f"**Web Search Usage:** {custom.get('web_search_usage', 0):.1f}%")
            st.write(f"**Tool Diversity:** {custom.get('tool_diversity', 0)} tools")


def display_comparison_results(comparison_dict):
    """Display model comparison results"""
    st.markdown("### 🔄 Model Comparison Results")
    
    results = comparison_dict.get('results', {})
    
    # Create comparison table
    import pandas as pd
    
    comparison_data = []
    for model_id, metrics in results.items():
        comparison_data.append({
            'Model': model_id.split(':')[1] if ':' in model_id else model_id,
            'Accuracy': f"{metrics.get('accuracy_score', 0):.2f}/10" if metrics.get('accuracy_score') else "N/A",
            'Reliability': f"{metrics.get('reliability_score', 0):.1f}%" if metrics.get('reliability_score') else "N/A",
            'Avg Latency': f"{metrics.get('performance_metrics', {}).get('avg_latency_seconds', 0):.2f}s" if metrics.get('performance_metrics') else "N/A"
        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Best performer analysis
        st.markdown("#### 🏆 Best Performers")
        
        # Find best accuracy
        best_accuracy_model = max(results.items(), 
                                key=lambda x: x[1].get('accuracy_score', 0) if x[1].get('accuracy_score') else 0)
        st.write(f"**Best Accuracy:** {best_accuracy_model[0]} ({best_accuracy_model[1].get('accuracy_score', 0):.2f}/10)")
        
        # Find best reliability  
        best_reliability_model = max(results.items(),
                                   key=lambda x: x[1].get('reliability_score', 0) if x[1].get('reliability_score') else 0)
        st.write(f"**Best Reliability:** {best_reliability_model[0]} ({best_reliability_model[1].get('reliability_score', 0):.1f}%)")


def create_sample_evaluation_dataset():
    """Create and download a sample evaluation dataset"""
    sample_dataset = {
        "questions": [
            "What is retrieval-augmented generation?",
            "How does vector search work in RAG systems?", 
            "What are the benefits of using Qdrant for vector storage?",
            "How do you evaluate RAG system performance?",
            "What is the difference between dense and sparse vectors?"
        ],
        "contexts": [
            "RAG combines retrieval and generation to provide accurate, context-aware responses",
            "Vector search finds similar documents using embedding similarity in high-dimensional space",
            "Qdrant provides fast vector search with filtering, hybrid search, and cloud deployment options", 
            "RAG evaluation involves measuring retrieval accuracy, generation quality, and overall system performance",
            "Dense vectors capture semantic meaning while sparse vectors focus on keyword matching"
        ],
        "expected_answers": [
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation to provide more accurate and contextual responses.",
            "Vector search works by converting documents and queries into high-dimensional vectors and finding the most similar ones using distance metrics.",
            "Qdrant offers benefits like high performance, hybrid search capabilities, easy scaling, and both cloud and on-premise deployment options.",
            "RAG systems are evaluated using metrics like retrieval accuracy, generation quality, faithfulness, relevance, and performance benchmarks.",
            "Dense vectors represent semantic meaning in continuous space while sparse vectors use discrete features for keyword-based matching."
        ],
        "ground_truths": [
            "RAG combines retrieval and generation for enhanced AI responses",
            "Vector similarity search using embeddings in multidimensional space", 
            "High-performance vector database with hybrid search and scalability",
            "Multi-dimensional evaluation including accuracy and performance metrics",
            "Semantic dense vectors vs keyword-focused sparse vectors"
        ]
    }
    
    return json.dumps(sample_dataset, indent=2)


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>EnterpriseGPT</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Your very own intelligent multimodal RAG assistant powered by Agno </p>",
        unsafe_allow_html=True,
    )

    ####################################################################
    # Model selector (temporarily limited to working models)
    ####################################################################
    model_options = {
        "o3-mini": "openai:o3-mini",
        "gpt-4o": "openai:gpt-4o",
        # Temporarily disabled due to compatibility issues
        # "gemini-2.0-flash-exp": "google:gemini-2.0-flash-exp",
        # "claude-3-5-sonnet": "anthropic:claude-3-5-sonnet-20241022", 
        # "llama-3.3-70b": "groq:llama-3.3-70b-versatile",
    }
    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    model_id = model_options[selected_model]

    ####################################################################
    # Initialize Agent
    ####################################################################
    agentic_rag_agent: Agent
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or st.session_state.get("current_model") != model_id
    ):
        logger.info("---*--- Creating new Agentic RAG  ---*---")
        agentic_rag_agent = get_agentic_rag_agent(model_id=model_id)
        st.session_state["agentic_rag_agent"] = agentic_rag_agent
        st.session_state["current_model"] = model_id
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    ####################################################################
    # Load Agent Session from the database
    ####################################################################
    # Check if session ID is already in session state
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
            # Continue anyway instead of returning, to avoid breaking session switching
    elif (
        st.session_state["agentic_rag_agent_session_id"]
        and hasattr(agentic_rag_agent, "memory")
        and agentic_rag_agent.memory is not None
        and not agentic_rag_agent.memory.runs
    ):
        # If we have a session ID but no runs, try to load the session explicitly
        try:
            agentic_rag_agent.load_session(
                st.session_state["agentic_rag_agent_session_id"]
            )
        except Exception as e:
            logger.error(f"Failed to load existing session: {str(e)}")
            # Continue anyway

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
            # Check if _run is an object with message attribute
            if hasattr(_run, "message") and _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            # Check if _run is an object with response attribute
            if hasattr(_run, "response") and _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)
    elif len(agent_runs) == 0 and len(st.session_state["messages"]) == 0:
        logger.debug("No run history found")

    if prompt := st.chat_input("👋 Ask me anything!"):
        add_message("user", prompt)

    ####################################################################
    # Track loaded URLs and files in session state
    ####################################################################
    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = set()
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = set()
    if "knowledge_base_initialized" not in st.session_state:
        st.session_state.knowledge_base_initialized = False

    st.sidebar.markdown("#### 📚 Document Management")
    input_url = st.sidebar.text_input("Add URL to Knowledge Base")
    if (
        input_url and not prompt and not st.session_state.knowledge_base_initialized
    ):  # Only load if KB not initialized
        if input_url not in st.session_state.loaded_urls:
            alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
            if input_url.lower().endswith(".pdf"):
                try:
                    # Download PDF to temporary file
                    response = requests.get(input_url, stream=True, verify=False)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name

                    reader = PDFReader()
                    docs: List[Document] = reader.read(tmp_path)

                    # Clean up temporary file
                    os.unlink(tmp_path)
                except Exception as e:
                    st.sidebar.error(f"Error processing PDF: {str(e)}")
                    docs = []
            else:
                scraper = WebsiteReader(max_links=2, max_depth=1)
                docs: List[Document] = scraper.read(input_url)

            if docs:
                agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_urls.add(input_url)
                st.sidebar.success("URL added to knowledge base")
            else:
                st.sidebar.error("Could not process the provided URL")
            alert.empty()
        else:
            st.sidebar.info("URL already loaded in knowledge base")

    uploaded_file = st.sidebar.file_uploader(
        "Add a Document (.pdf, .csv, or .txt)", key="file_upload"
    )
    if (
        uploaded_file and not prompt and not st.session_state.knowledge_base_initialized
    ):  # Only load if KB not initialized
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in st.session_state.loaded_files:
            alert = st.sidebar.info("Processing document...", icon="ℹ️")
            file_type = uploaded_file.name.split(".")[-1].lower()
            reader = get_reader(file_type)
            if reader:
                docs = reader.read(uploaded_file)
                agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_files.add(file_identifier)
                st.sidebar.success(f"{uploaded_file.name} added to knowledge base")
                st.session_state.knowledge_base_initialized = True
            alert.empty()
        else:
            st.sidebar.info(f"{uploaded_file.name} already loaded in knowledge base")

    if st.sidebar.button("Clear Knowledge Base"):
        # For Qdrant, we delete the collection instead of the database
        agentic_rag_agent.knowledge.vector_db.delete()
        st.session_state.loaded_urls.clear()
        st.session_state.loaded_files.clear()
        st.session_state.knowledge_base_initialized = False  # Reset initialization flag
        st.sidebar.success("Knowledge base cleared")
    
    ###############################################################
    # Sample Question
    ###############################################################
    st.sidebar.markdown("#### ❓ Sample Questions")
    if st.sidebar.button("📝 Summarize"):
        add_message(
            "user",
            "Can you summarize what is currently in the knowledge base (use `search_knowledge_base` tool)?",
        )

    ###############################################################
    # Evaluation Interface
    ###############################################################
    run_evaluation_interface()
    
    # Sample dataset download
    if EVALUATION_AVAILABLE:
        st.sidebar.markdown("#### 📋 Sample Dataset")
        if st.sidebar.download_button(
            "📥 Download Sample Dataset",
            create_sample_evaluation_dataset(),
            file_name="sample_rag_evaluation_dataset.json",
            mime="application/json",
            key="download_sample_dataset"
        ):
            st.sidebar.success("📊 Sample dataset downloaded!")

    ###############################################################
    # Utility buttons
    ###############################################################
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])  # Equal width columns
    with col1:
        if st.sidebar.button(
            "🔄 New Chat", use_container_width=True
        ):  # Added use_container_width
            restart_agent()
    with col2:
        if st.sidebar.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="rag_chat_history.md",
            mime="text/markdown",
            use_container_width=True,  # Added use_container_width
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
                    # Display tool calls if they exist in the message
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
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = agentic_rag_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        # Display response
                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message(
                        "assistant", response, agentic_rag_agent.run_response.tools
                    )
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)

    ####################################################################
    # Session selector
    ####################################################################
    session_selector_widget(agentic_rag_agent, model_id)
    rename_session_widget(agentic_rag_agent)

    ####################################################################
    # About section
    ####################################################################
    about_widget()


main()