# ==========================================================
# app.py
# Customer Journey Academy KT Assistant
# ChatGPT-style Streamlit App with RAG + GPT Streaming
# ==========================================================

import os
import io
import json
import time
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import logging
import sqlite3
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
import threading

# ==========================================================
# OPTIONAL IMPORTS
# ==========================================================

try:
    from openai import OpenAI
except:
    OpenAI = None

# We defer loading LangChain/FAISS/HuggingFace heavy objects until needed.
FAISS = None
PyPDFLoader = None
RecursiveCharacterTextSplitter = None

vector_db_status = {
    "building": False,
    "loading": False,
    "ready": False
}

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Customer Journey KT Assistant",
    page_icon="🚀",
    layout="wide"
)

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

DOCS_DIR = BASE_DIR / "docs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# ==========================================================
# APP CONFIG
# ==========================================================

TEAM_NAME = "Customer Journey Academy"

from dataclasses import dataclass
import os

@dataclass
class Config:

    MODEL = os.getenv(
        "MODEL",
        "gpt-5-mini"
    )

    DEFAULT_MODEL = "gpt-5-mini"

    TEMPERATURE = float(
        os.getenv(
            "TEMPERATURE",
            "0.3"
        )
    )

    TOP_K = int(
        os.getenv(
            "TOP_K",
            "5"
        )
    )

    CHUNK_SIZE = 1000

    CHUNK_OVERLAP = 150

    MAX_HISTORY = 30

SUPPORTED_MODELS = [
    "gpt-5-mini",
    "gpt-5.5"
]

# backward-compatible default
DEFAULT_MODEL = Config.DEFAULT_MODEL

SYSTEM_PROMPT = """
You are Comci, the Customer Journey Academy KT Assistant.

You are an onboarding mentor.

Always:

1. Explain thoroughly.
2. Use numbered sections.
3. Highlight important terms in bold.
4. Mention common mistakes.
5. Reference source documents and sections.
6. Suggest the next topic to learn.
7. Use markdown formatting.
"""

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s %(levelname)s %(message)s"

)

logger = logging.getLogger(
    "Comci"
)
# ==========================================================
# DOCUMENT CONFIG
# ==========================================================

KT_DOCS = [
    {
        "name": "Introduction",
        "file": "Introdoc.pdf",
        "category": "Getting Started",
        "icon": "📘",
        "badge": "core"
    },
    {
        "name": "Assisted Journey",
        "file": "Assistedjourney.pdf",
        "category": "Customer Journeys",
        "icon": "🤝",
        "badge": "core"
    },
    {
        "name": "Upselling Journey",
        "file": "Upselljourney.pdf",
        "category": "Customer Journeys",
        "icon": "🚀",
        "badge": "updated"
    },
    {
        "name": "Billing Journey",
        "file": "Billingjourney.pdf",
        "category": "Operations",
        "icon": "💳",
        "badge": "core"
    }
]

# ==========================================================
# LEARNING PATH
# ==========================================================

LEARNING_PATH = [
    {
        "step": 1,
        "title": "Introduction",
        "status": "done",
        "desc": "Understand onboarding and fundamentals."
    },
    {
        "step": 2,
        "title": "Assisted Journey",
        "status": "current",
        "desc": "Customer support workflows."
    },
    {
        "step": 3,
        "title": "Upselling Journey",
        "status": "pending",
        "desc": "Customer growth workflows."
    },
    {
        "step": 4,
        "title": "Billing Journey",
        "status": "pending",
        "desc": "Billing operations."
    }
]

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>

.block-container{
padding-top:1rem;
}

:root{
--purple:#534AB7;
--green:#1D9E75;
--light:#EEEDFE;
}

.hero{
background:#534AB7;
padding:24px;
border-radius:18px;
color:white;
text-align:center;
margin-bottom:20px;
}

.ref-card{
border:1px solid #DDD;
border-radius:14px;
padding:12px;
margin-bottom:10px;
background:white;
}

.badge-core{
background:#534AB7;
color:white;
padding:4px 10px;
border-radius:20px;
}

.badge-updated{
background:#1D9E75;
color:white;
padding:4px 10px;
border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🚀 Customer Journey KT Assistant</h1>
<p>AI Powered Knowledge Transfer Assistant</p>
</div>
""", unsafe_allow_html=True)

##
# ==========================================================
# SQLITE
# ==========================================================

conn = sqlite3.connect(

    "chat_history.db",

    check_same_thread=False

)

cursor = conn.cursor()

cursor.execute(
"""
CREATE TABLE IF NOT EXISTS conversations(

id INTEGER PRIMARY KEY,

title TEXT,

chat TEXT,

created_at TEXT

)
"""
)

conn.commit()
# ==========================================================
# SESSION
# ==========================================================

WELCOME = """
👋 Hi, I'm Comci.

I can help with:

📘 Introduction

🤝 Assisted Journey

🚀 Upselling Journey

💳 Billing Journey

Ask me anything.
"""

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME,
            "references": []
        }
    ]

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

if "seen_welcome" not in st.session_state:
    st.session_state.seen_welcome = False
    
if "tokens" not in st.session_state:

    st.session_state.tokens = 0    

if "last_prompt" not in st.session_state:

    st.session_state.last_prompt = ""

# conversation pruning when history grows too large
if len(st.session_state.messages) > getattr(Config, "MAX_HISTORY", 0):

    st.session_state.messages = (

        st.session_state.messages[:5]

        +

        st.session_state.messages[-20:]

    )

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("⚙ Settings")

    st.session_state.model = st.selectbox(
        "Model",
        SUPPORTED_MODELS
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.3
    )

    st.divider()

    if st.button("🧹 New Chat"):

        st.session_state.messages = [
            {
                "role":"assistant",
                "content":WELCOME,
                "references":[],
                "show_mindmap":False
            }
        ]

        st.rerun()

    st.divider()

    export_data = json.dumps(
        st.session_state.messages,
        indent=2
    )

    st.download_button(
        "📥 Export Chat",
        export_data,
        file_name="conversation.json"
    )
    st.divider()

st.subheader(
    "History"
)

cursor.execute(
"""
SELECT id,title
FROM conversations
ORDER BY id DESC
LIMIT 20
"""
)

rows = cursor.fetchall()

for row in rows:

    st.caption(
        row[1]
    )

st.metric(

    "Tokens",

    st.session_state.tokens

)    

# ==========================================================
# OPENAI CLIENT
# ==========================================================

def init_client():

    if OpenAI is None:
        return None

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_PAI_KEY")
        or os.getenv("OPEN_API_KEY")
    )

    if not api_key:
        return None

    return OpenAI(api_key=api_key)

# ==========================================================
# PART 1 END
# ==========================================================
# ==========================================================
# VECTOR DB + PDF INGESTION + RETRIEVAL + GPT STREAMING
# PART 2
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():

    try:

        from langchain_huggingface import HuggingFaceEmbeddings

    except Exception:

        return None

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================================
# LOAD PDFS
# ==========================================================

def load_documents():

    documents = []

    if not DOCS_DIR.exists():
        return documents

    try:

        from langchain_community.document_loaders import PyPDFLoader

    except Exception:

        return documents

    pdfs = list(DOCS_DIR.glob("*.pdf"))

    for pdf in pdfs:

        try:

            loader = PyPDFLoader(str(pdf))

            docs = loader.load()

            for d in docs:

                d.metadata["source"] = pdf.name

                if "page" not in d.metadata:
                    d.metadata["page"] = 1

            documents.extend(docs)

        except Exception:

            pass

    return documents


# ==========================================================
# SPLIT DOCUMENTS
# ==========================================================

def split_documents(documents):

    try:

        from langchain_text_splitters import RecursiveCharacterTextSplitter

    except Exception:

        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


# ==========================================================
# BUILD VECTOR DB
# ==========================================================

def build_vector_db():

    embeddings = get_embeddings()

    if embeddings is None:
        return None

    docs = load_documents()

    if len(docs) == 0:
        return None

    chunks = split_documents(docs)

    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        return None

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTOR_DB_DIR.mkdir(
        exist_ok=True
    )

    db.save_local(
        str(VECTOR_DB_DIR)
    )

    return db


# ==========================================================
# LOAD VECTOR DB
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_vector_db():

    embeddings = get_embeddings()

    if embeddings is None:
        return None

    try:

        if (
            VECTOR_DB_DIR.exists()
            and
            (VECTOR_DB_DIR / "index.faiss").exists()
        ):

            try:
                from langchain_community.vectorstores import FAISS
            except Exception:
                return build_vector_db()

            db = FAISS.load_local(
                str(VECTOR_DB_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )

            return db

        return build_vector_db()

    except Exception:

        return build_vector_db()


def _set_vector_db_ready():

    vector_db_status["ready"] = True

    vector_db_status["building"] = False

    vector_db_status["loading"] = False


def _async_build_vector_db():

    try:

        build_vector_db()

        if VECTOR_DB_DIR.exists() and (VECTOR_DB_DIR / "index.faiss").exists():

            _set_vector_db_ready()

    except Exception as e:

        logger.error(f"Vector DB build failed: {e}")

    finally:

        vector_db_status["building"] = False


def _async_load_vector_db():

    try:

        load_vector_db()

        if VECTOR_DB_DIR.exists() and (VECTOR_DB_DIR / "index.faiss").exists():

            _set_vector_db_ready()

    except Exception as e:

        logger.error(f"Vector DB load failed: {e}")

    finally:

        vector_db_status["loading"] = False


def start_vector_db_background():

    index_path = VECTOR_DB_DIR / "index.faiss"

    if VECTOR_DB_DIR.exists() and index_path.exists():

        if not vector_db_status["ready"] and not vector_db_status["loading"]:

            vector_db_status["loading"] = True

            threading.Thread(target=_async_load_vector_db, daemon=True).start()

        return False

    if not vector_db_status["building"]:

        vector_db_status["building"] = True

        threading.Thread(target=_async_build_vector_db, daemon=True).start()

    return False


def ensure_vector_db_background():

    if vector_db_status["ready"]:

        return True

    return start_vector_db_background()


def get_db():

    """Lazy accessor for the vector DB."""

    if vector_db_status["ready"]:

        try:

            return load_vector_db()

        except Exception:

            return None

    return None


# ==========================================================
# RETRIEVE DOCUMENTS
# ==========================================================

def retrieve_docs(query, k=5):

    if not vector_db_status["ready"]:

        ensure_vector_db_background()

        return [], []

    db = get_db()

    if db is None:

        return [], []

    try:

        docs = db.max_marginal_relevance_search(
            query,
            k=Config.TOP_K,
            fetch_k=20,
    
        )

        refs = []

        context = []

        for d in docs:

            context.append(
                d.page_content
            )

            refs.append(
                {
                    "doc":
                    d.metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "section":
                    f"Page {d.metadata.get('page',1)}"
                }
            )

        return context, refs

    except Exception:

        return [], []


# ==========================================================
# GPT STREAM
# ==========================================================
@retry(

    stop=stop_after_attempt(3),

    wait=wait_exponential()

)
def stream_answer(question):

    client = init_client()

    if client is None:

        yield """
OpenAI API Key not found.

Set:

OPENAI_API_KEY

and restart Streamlit.
"""

        return

    context, refs = retrieve_docs(
        question
    )

    rag_context = "\n\n".join(
        context
    )

    messages = [

        {
            "role":"system",
            "content":SYSTEM_PROMPT
        }

    ]

    # only include the most recent messages to reduce request size
    history = st.session_state.messages
    try:
        history = history[-Config.MAX_HISTORY:]
    except Exception:
        history = history

    for m in history:

        messages.append(
            {
                "role": m["role"],
                "content": m["content"]
            }
        )

    messages.append(
        {
            "role":"user",
            "content":
f"""
Question:

{question}

Relevant KT Content:

{rag_context}
"""
        }
    )

    try:

        stream = client.chat.completions.create(

            model=st.session_state.model,

            temperature=st.session_state.temperature,

            messages=messages,

            stream=True

        )

        for chunk in stream:

            try:

                delta = chunk.choices[0].delta

                if delta.content:

                    yield delta.content

            except:

                pass

    except Exception as e:

        yield f"Error: {e}"


# ==========================================================
# GENERATE FULL ANSWER
# ==========================================================

def generate_answer(question):

    if not vector_db_status["ready"]:

        ensure_vector_db_background()

        start = time.time()

        while not vector_db_status["ready"]:

            time.sleep(0.5)

            if time.time() - start > 120:

                yield (
                    "Preparing the knowledge base is still in progress. "
                    "Please wait a moment and try again."
                ), []

                return

    text = ""

    refs = []

    for piece in stream_answer(question):

        text += piece

        yield text, refs


# ==========================================================
# KEY CHECK
# ==========================================================

def api_key_status():

    client = init_client()

    if client is None:

        return (
            False,
            "OPENAI_API_KEY not configured"
        )

    # Avoid making a blocking network call at startup (models.list()).
    # Presence of a client is a lightweight indicator the key is configured.
    return (
        True,
        "Connected"
    )


# ==========================================================
# STARTUP STATUS
# ==========================================================

if "api_status" not in st.session_state:

    ok, msg = api_key_status()

    st.session_state.api_status = ok

    st.session_state.api_msg = msg


# ==========================================================
# PART 2 END
# ==========================================================

# ==========================================================
# ORDER JOURNEY MIND MAP
# ==========================================================

MINDMAP_PATH = BASE_DIR / "components" / "skyorderjourney.html"

def is_order_journey_query(question: str) -> bool:
    """Return True when the user is explicitly asking about the Sky Q order journey."""
    q = " ".join((question or "").lower().split())

    triggers = (
        "order journey",
        "order flow",
        "order process",
        "sky q flow",
        "sky q journey",
        "digital journey",
        "explain the order",
        "order creation flow",
        "order lifecycle",
    )

    return any(trigger in q for trigger in triggers)


def render_order_journey_mindmap(height: int = 850):
    """Render the existing interactive Sky Q order journey HTML."""
    if not MINDMAP_PATH.exists():
        st.warning(
            "🗺️ The Order Journey mind map could not be found. "
            "Please make sure components/skyorderjourney.html exists."
        )
        return

    try:
        html = MINDMAP_PATH.read_text(encoding="utf-8")
        components.html(
            html,
            height=height,
            scrolling=True,
        )
    except Exception as e:
        logger.error(f"Order Journey mind map rendering failed: {e}")
        st.error("Unable to load the Order Journey mind map.")


def save_chat():

    try:

        chat = json.dumps(st.session_state.messages)

        cursor.execute(
            """
            INSERT INTO conversations(title, chat, created_at)
            VALUES(?,?,?)
            """,
            ("Conversation", chat, str(datetime.now()))
        )

        conn.commit()

    except Exception as e:

        logger.error(str(e))


# ==========================================================
# PART 3
# CHATGPT STYLE UI + STREAMING + REFERENCES
# ==========================================================

# ----------------------------------------------------------
# API STATUS
# ----------------------------------------------------------

if st.session_state.api_status:

    st.success(
        f"✅ OpenAI Connected ({st.session_state.api_msg})"
    )

else:

    st.warning(
        st.session_state.api_msg
    )

# ==========================================================
# TABS
# ==========================================================

chat_tab, docs_tab, learning_tab = st.tabs(
    [
        "💬 Chat",
        "📚 KT Docs",
        "🛣 Learning Path"
    ]
)

# ==========================================================
# CHAT TAB
# ==========================================================

with chat_tab:

    # ------------------------------------------------------
    # Welcome popup
    # ------------------------------------------------------

    if not st.session_state.seen_welcome:

        try:

            with st.modal(
                "Welcome to Customer Journey Academy"
            ):

                st.markdown(
                    """
# 👋 Hi, I'm Comci

I'm your KT Assistant.

I can help you with:

- 📘 Introduction
- 🤝 Assisted Journey
- 🚀 Upselling Journey
- 💳 Billing Journey

Ask me anything.
"""
                )

                if st.button(
                    "Get Started"
                ):

                    st.session_state.seen_welcome = True

                    st.rerun()

        except:

            st.info(
                "👋 Hi, I'm Comci. Thanks for choosing me as a KT assistant"
            )

    # ------------------------------------------------------
    # Conversation
    # ------------------------------------------------------

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):

            st.markdown(
                msg["content"]
            )

            if msg.get("show_mindmap", False):
                st.markdown("### 🗺️ Interactive Sky Q Sat Digital Journey")
                render_order_journey_mindmap()

            refs = msg.get(
                "references",
                []
            )

            if refs:

                st.markdown(
                    "### 📚 References"
                )

                for ref in refs:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"""
**Document**

{ref['doc']}

**Section**

{ref['section']}
"""
                        )

    # ------------------------------------------------------
    # Input
    # ------------------------------------------------------

    prompt = st.chat_input(
        "Ask about onboarding, billing, journeys..."
    )

    if prompt:

        show_mindmap = is_order_journey_query(prompt)

        st.session_state.messages.append(
            {
                "role":"user",
                "content":prompt,
                "references":[]
            }
        )

        # store last prompt for regenerate
        st.session_state.last_prompt = prompt

        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )

        # --------------------------------------------------
        # Assistant
        # --------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            placeholder = st.empty()

            response_text = ""

            refs = []

            start_time = time.time()

            if not vector_db_status["ready"]:
                placeholder.markdown(
                    "⏳ Preparing the knowledge base..."
                )

            spinner_msg = "Preparing the knowledge base..." if not vector_db_status["ready"] else "Thinking..."
            with st.spinner(spinner_msg):

                for partial, refs in generate_answer(
                    prompt
                ):

                    response_text = partial

                    placeholder.markdown(
                        response_text + "▌"
                    )

                placeholder.markdown(
                    response_text
                )

            elapsed = time.time() - start_time

            st.caption(f"Response time: {elapsed:.2f}s")

            # copy/download response
            st.download_button(
                "📋 Copy",
                response_text,
                file_name="response.txt"
            )

            # ----------------------------------------------
            # references
            # ----------------------------------------------

            if refs:

                st.markdown(
                    "### 📚 Referenced KT Documents"
                )

                for ref in refs:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"""
**Document**

{ref['doc']}

**Section**

{ref['section']}
"""
                        )

            if show_mindmap:
                st.markdown("### 🗺️ Interactive Sky Q Sat Digital Journey")
                render_order_journey_mindmap()

            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":response_text,
                    "references":refs,
                    "show_mindmap":show_mindmap
                }
            )

            # persist conversation
            try:
                save_chat()
            except Exception:
                pass

            st.rerun()

# ==========================================================
# CHAT EXPORT
# ==========================================================

def export_chat_markdown():

    md = "# Conversation\n\n"

    for m in st.session_state.messages:

        role = m["role"].upper()

        md += (
            f"## {role}\n\n"
            f"{m['content']}\n\n"
        )

    return md


with st.sidebar:

    st.divider()

    st.download_button(

        "📄 Export Markdown",

        export_chat_markdown(),

        file_name="conversation.md"

    )

# ==========================================================
# CLEAR CHAT
# ==========================================================

with st.sidebar:

    if st.button(
        "🗑 Clear History"
    ):

        st.session_state.messages = [

            {
                "role":"assistant",
                "content":WELCOME,
                "references":[],
                "show_mindmap":False
            }

        ]

        st.rerun()

# ==========================================================
# CHAT STATS
# ==========================================================

with st.sidebar:

    st.divider()

    st.caption(
        f"Messages: {len(st.session_state.messages)}"
    )

    st.caption(
        f"Model: {st.session_state.model}"
    )

# ==========================================================
# PART 3 END
# ==========================================================
# ==========================================================
# PART 4
# DOCS TAB + LEARNING PATH + FOOTER
# ==========================================================

# ==========================================================
# DOCS TAB
# ==========================================================

with docs_tab:

    st.subheader(
        "📚 KT Document Library"
    )

    search = st.text_input(
        "Search documents"
    )

    grouped_docs = {}

    for doc in KT_DOCS:

        if search:

            if search.lower() not in doc["name"].lower():

                continue

        grouped_docs.setdefault(
            doc["category"],
            []
        ).append(doc)

    if not grouped_docs:

        st.info(
            "No matching documents."
        )

    for category, docs in grouped_docs.items():

        st.markdown(
            f"## {category}"
        )

        for doc in docs:

            path = DOCS_DIR / doc["file"]

            col1, col2, col3 = st.columns(
                [6,2,2]
            )

            with col1:

                st.markdown(
                    f"{doc['icon']} **{doc['name']}**"
                )

                st.caption(
                    str(path)
                )

            with col2:

                if doc["badge"] == "core":

                    st.markdown(
                        "<span class='badge-core'>core</span>",
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        "<span class='badge-updated'>updated</span>",
                        unsafe_allow_html=True
                    )

            with col3:

                if path.exists():

                    try:

                        with open(
                            path,
                            "rb"
                        ) as f:

                            st.download_button(
                                "📥 Download",
                                data=f.read(),
                                file_name=doc["file"],
                                key=f"dl_{doc['file']}"
                            )

                    except:

                        pass


# ==========================================================
# LEARNING TAB
# ==========================================================

with learning_tab:

    st.subheader(
        "🛣 New Joiner Learning Path"
    )

    completed = len(
        [
            x
            for x in LEARNING_PATH
            if x["status"] == "done"
        ]
    )

    progress = completed / len(
        LEARNING_PATH
    )

    st.progress(
        progress
    )

    st.caption(
        f"{int(progress*100)}% Complete"
    )

    for step in LEARNING_PATH:

        icon = "⚪"

        if step["status"] == "done":

            icon = "✅"

        elif step["status"] == "current":

            icon = "🟣"

        with st.container(
            border=True
        ):

            st.markdown(
                f"""
### {icon} Step {step['step']}

## {step['title']}

{step['desc']}
"""
            )

            if st.button(

                f"Explain {step['title']}",

                key=f"step_{step['step']}"

            ):

                question = (
                    f"Explain {step['title']}"
                )

                st.session_state.messages.append(

                    {
                        "role":"user",
                        "content":question,
                        "references":[]
                    }

                )

                st.rerun()

# ==========================================================
# SIDEBAR RETRIEVAL SETTINGS
# ==========================================================

with st.sidebar:

    st.divider()

    st.subheader(
        "🔎 Retrieval"
    )

    retrieval_k = st.slider(

        "Top Documents",

        1,

        10,

        5

    )

    # Build vector DB on demand (non-blocking)
    if st.button("Build Vector DB"):

        if vector_db_status["ready"]:

            st.success("Vector DB is already ready.")

        elif vector_db_status["building"] or vector_db_status["loading"]:

            st.info("Vector DB preparation is already in progress.")

        else:

            start_vector_db_background()

            st.info("Started building vector DB in background.")

# ==========================================================
# STARTUP INFO
# ==========================================================

with st.sidebar:

    st.divider()

    st.caption(
        "🚀 Customer Journey Academy"
    )

    st.caption(
        "Comci KT Assistant"
    )

    st.caption(
        f"Time: {datetime.now().strftime('%H:%M')}"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🚀 Customer Journey Academy KT Assistant"
)

st.caption(
    "Built with Streamlit + OpenAI + LangChain + FAISS"
)

# ==========================================================
# OPTIONAL LOADING STATUS
# ==========================================================

try:

    with st.sidebar:

        st.divider()

        if VECTOR_DB_DIR.exists():

            st.success(
                "Vector DB Ready"
            )

        else:

            st.warning(
                "Vector DB will be created automatically"
            )

except:

    pass

with st.sidebar:

    if st.button("↻ Regenerate") and st.session_state.last_prompt:

        text = ""

        for piece, _ in generate_answer(st.session_state.last_prompt):

            text = piece

        st.session_state.messages.extend(
            [
                {
                    "role": "user",
                    "content": st.session_state.last_prompt,
                    "references": []
                },
                {
                    "role": "assistant",
                    "content": text,
                    "references": []
                }
            ]
        )

        save_chat()
        st.rerun()
# ==========================================================
# END OF app.py
# ==========================================================