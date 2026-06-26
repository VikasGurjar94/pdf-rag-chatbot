from dotenv import load_dotenv

load_dotenv()

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

PDF_PATH = "data/data.pdf"

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
# Load Documents
# ============================================================

def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()


# ============================================================
# Split Documents
# ============================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


# ============================================================
# Embedding Model
# ============================================================

def create_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# Vector Store
# ============================================================

def create_vector_store(chunks, embeddings):
    vector_store = InMemoryVectorStore(embeddings)

    vector_store.add_documents(chunks)

    return vector_store


# ============================================================
# LLM
# ============================================================

def create_llm():
    return ChatMistralAI(
        model=LLM_MODEL,
        max_retries=6,
    )


# ============================================================
# Context Retrieval
# ============================================================

def retrieve_context(retriever, query):
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context


# ============================================================
# Main
# ============================================================

def main():

    print("Loading documents...")

    documents = load_documents()

    chunks = split_documents(documents)

    embeddings = create_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    model = create_llm()

    conversation = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    print("\nRAG Chat Started")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("You : ")

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        context = retrieve_context(
            retriever,
            query,
        )

        conversation.append(
            HumanMessage(
                content=f"""
Context:

{context}

Question:

{query}
"""
            )
        )

        response = model.invoke(conversation)

        conversation.append(
            AIMessage(content=response.content)
        )

        print("\nAssistant:\n")
        print(response.content)
        print("-" * 70)


if __name__ == "__main__":
    main()