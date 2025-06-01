# Agentic RAG Agent

**Agentic RAG Agent** is a chat application that combines models with retrieval-augmented generation.
It allows users to ask questions based on custom knowledge bases, documents, and web data, retrieve context-aware answers, and maintain chat history across sessions.

> Note: Fork and clone this repository if needed

### 1. Create a virtual environment

```shell
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```shell
pip install -r cookbook/examples/streamlit_apps/agentic_rag/requirements.txt
```

### 3. Configure API Keys

Required:
```bash
export OPENAI_API_KEY=your_openai_key_here
```

Optional (for additional models):
```bash
export ANTHROPIC_API_KEY=your_anthropic_key_here
export GOOGLE_API_KEY=your_google_key_here
export GROQ_API_KEY=your_groq_key_here
```

Optional (for Qdrant Cloud):
```bash
export QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
export QDRANT_API_KEY=your_qdrant_api_key_here
```

### 4. Run Qdrant

You have several options to run Qdrant:

#### Option A: Run Qdrant locally with Docker

> Install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) first.

```shell
docker run -p 6333:6333 qdrant/qdrant
```

#### Option B: Run Qdrant with persistent storage

```shell
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

#### Option C: Use Qdrant Cloud

1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster
3. Set the environment variables:
   ```bash
   export QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
   export QDRANT_API_KEY=your_api_key_here
   ```

### 5. Run PgVector (for session storage)

> Install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) first.

Session data is still stored in PostgreSQL. You can run it using:

- Run using a helper script

```shell
./cookbook/scripts/run_pgvector.sh
```

- OR run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

### 6. Run Agentic RAG App

```shell
streamlit run cookbook/examples/streamlit_apps/agentic_rag/app.py 
```

## 🔧 Customization

### Model Selection

The application supports multiple model providers:
- OpenAI (o3-mini, gpt-4o)
- Anthropic (claude-3-5-sonnet)
- Google (gemini-2.0-flash-exp)
- Groq (llama-3.3-70b-versatile)

### Vector Database Configuration

The application now uses Qdrant for vector storage:
- **Local Qdrant**: Default configuration (http://localhost:6333)
- **Qdrant Cloud**: Configure via environment variables
- **Session Storage**: Still uses PostgreSQL for agent session persistence

### How to Use
- Open [localhost:8501](http://localhost:8501) in your browser.
- Upload documents or provide URLs (websites, csv, txt, and PDFs) to build a knowledge base.
- Enter questions in the chat interface and get context-aware answers.
- The app can also answer question using duckduckgo search without any external documents added.

### Troubleshooting
- **Qdrant Connection Issues**: Ensure Qdrant is running on port 6333 (`docker ps`).
- **PostgreSQL Connection Refused**: Ensure `pgvector` container is running for session storage.
- **OpenAI API Errors**: Verify that the `OPENAI_API_KEY` is set and valid.
- **Qdrant Cloud Issues**: Check your `QDRANT_URL` and `QDRANT_API_KEY` are correctly set.

## 📚 Documentation

For more detailed information:
- [Agno Documentation](https://docs.agno.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

## 🤝 Support

Need help? Join our [Discord community](https://agno.link/discord)

# enterprise-multimodal-rag