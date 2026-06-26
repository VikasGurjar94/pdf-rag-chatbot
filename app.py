from dotenv import load_dotenv

load_dotenv()

import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "mistral-large-latest"

TOP_K = 3

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Your task is to answer the user's question ONLY using the provided context.

Rules:

- Use the retrieved context as your primary source of truth.
- If the answer exists in the context, answer clearly.
- If only partial information is available, mention that.
- If the answer is not present in the context, reply:
  "I couldn't find the answer in the provided document."
- Do not make up facts.
- Keep answers concise and natural.
"""

# ============================================================
# Functions
# ============================================================

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def create_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


def create_vector_store(chunks, embeddings):
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def create_llm():
    return ChatMistralAI(
        model=LLM_MODEL,
        max_retries=6,
    )


def retrieve_context(retriever, query):
    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Simple RAG Chat",
    layout="centered"
)

st.title("📄 Simple RAG Chat")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if uploaded_pdf is not None and st.session_state.retriever is None:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        documents = load_documents(pdf_path)

        chunks = split_documents(documents)

        embeddings = create_embedding_model()

        vector_store = create_vector_store(
            chunks,
            embeddings,
        )

        st.session_state.retriever = vector_store.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

        st.session_state.model = create_llm()

        st.session_state.conversation = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

    st.success("PDF Loaded Successfully!")

# Display Chat

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input

if st.session_state.retriever:

    prompt = st.chat_input("Ask anything about the PDF...")

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        context = retrieve_context(
            st.session_state.retriever,
            prompt
        )

        st.session_state.conversation.append(
            HumanMessage(
                content=f"""
Context:

{context}

Question:

{prompt}
"""
            )
        )

        response = st.session_state.model.invoke(
            st.session_state.conversation
        )

        st.session_state.conversation.append(
            AIMessage(
                content=response.content
            )
        )

        with st.chat_message("assistant"):
            st.markdown(response.content)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

else:
    st.info("Upload a PDF to start chatting.")