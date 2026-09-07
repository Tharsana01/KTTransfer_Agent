Comci: Customer Journey KT Assistant

Comci is an AI-powered knowledge-transfer assistant for the **Customer Journey Academy**. It helps new joiners learn customer-support and operations workflows through a Streamlit chat experience grounded in the team's PDF training material.

## Project Context

The application is designed to support onboarding across these knowledge areas:

- **Introduction**: onboarding fundamentals
- **Assisted Journey**: customer support workflows
- **Upselling Journey**: customer growth workflows
- **Billing Journey**: billing operations
- **Sky Q order journey**: an optional interactive digital journey mind map

The assistant is intended to answer questions from the internal KT documents, cite the source document and page, explain concepts in numbered sections, call out common mistakes, and suggest the next topic to learn.

## How It Works

1. PDF files are read from the `docs/` directory.
2. PDF pages are split into overlapping text chunks.
3. Hugging Face `all-MiniLM-L6-v2` embeddings are used to create a FAISS vector index.
4. The index is stored locally in `vector_db/` and reused on later runs.
5. Relevant chunks are retrieved for each question using maximal marginal relevance.
6. The retrieved content and recent conversation history are sent to the selected OpenAI model.
7. The response is streamed into the chat and displayed with document and page references.

If no vector index exists, Comci builds one in the background. The **Build Vector DB** control in the sidebar can also start this process manually.

## Main Features

- Streamlit chat interface with streamed responses
- Retrieval-augmented generation over local PDF documents
- Configurable OpenAI model and response temperature
- Conversation persistence in `chat_history.db`
- Recent conversation history shown in the sidebar
- Export conversations as JSON, Markdown, or text responses
- Knowledge-transfer document library with PDF downloads
- New-joiner learning path with progress tracking
- Optional interactive Sky Q order-journey mind map
- Background vector database loading and creation

## Project Structure

```text
COMCI/
├── app.py                         # Streamlit application and RAG pipeline
├── readme.md                      # Project context and setup notes
├── docs/                          # Source KT PDFs; create this directory locally
├── vector_db/                     # Generated FAISS index; created automatically
├── components/                    # Optional embedded UI components
│   └── sky_q_order_journey_mindmap.html
└── chat_history.db                # Local SQLite conversation history
```

The document filenames currently configured by the application are:

```text
docs/Introdoc.pdf
docs/Assistedjourney.pdf
docs/Upselljourney.pdf
docs/Billingjourney.pdf
```

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- The KT PDF files placed in `docs/`
- Optional: `components/sky_q_order_journey_mindmap.html` for the order-journey visualization

Install the Python dependencies:

```bash
pip install streamlit openai tenacity langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu pypdf sentence-transformers
```

## Configuration

Set the OpenAI API key before starting the application.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `gpt-5-mini` | Default model configuration |
| `TEMPERATURE` | `0.3` | Default response randomness |
| `TOP_K` | `5` | Number of retrieved document chunks |

The model and temperature can also be changed from the Streamlit sidebar.

## Run Locally

From the project directory:

```bash
streamlit run app.py
```

Streamlit will print the local URL, normally `http://localhost:8501`.

On first use, allow time for the embedding model to download and for the FAISS index to be created. Rebuild the index after adding or changing source PDFs.

## Data and Generated Files

- `docs/` contains the source knowledge-transfer material.
- `vector_db/` contains generated embeddings and should be rebuilt when documents change.
- `chat_history.db` stores conversations locally in SQLite.
- Exported conversations are generated from the current session and are not part of the source knowledge base.

Do not commit API keys or other secrets. The application reads the key from the `OPENAI_API_KEY` environment variable.

## Typical User Flow

1. Start the Streamlit application.
2. Confirm that the OpenAI key is detected in the sidebar status.
3. Wait for the knowledge base to become ready, or select **Build Vector DB**.
4. Ask a question about one of the KT journeys.
5. Review the streamed answer and its document/page references.
6. Use **KT Docs** and **Learning Path** to browse or continue onboarding.
