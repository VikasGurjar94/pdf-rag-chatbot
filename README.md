# 📄 Simple PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) project built using LangChain.

## Features

<!-- - PDF Loader -->
<!-- - Recursive Text Splitter -->
<!-- - Gemini Embeddings -->
<!-- - InMemory Vector Store -->
- Mistral Large LLM
- Context Retrieval
- Conversational Question Answering

## Tech Stack

- Python
- LangChain
- Gemini Embeddings
- Mistral AI
- PyPDF
- UV

## Installation

```bash
uv sync
```

Create a `.env`

```env
GOOGLE_API_KEY=YOUR_KEY
MISTRAL_API_KEY=YOUR_KEY
```

Run

```bash
python core.py
```

## Project Flow

```
PDF
↓
Loader
↓
Splitter
↓
Embedding
↓
Vector Store
↓
Retriever
↓
LLM
↓
Answer
```

## Author

Vikas Gurjar