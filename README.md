# DocMind

A local RAG-based PDF study assistant for asking questions and finding information from uploaded documents.

## Overview

DocMind is a Streamlit application that allows users to upload one or more PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant sections from uploaded documents and generate answers using locally running LLMs through Ollama.

The application runs locally and does not require an external AI API.

## Features

- Upload and work with multiple PDF documents
- Ask questions about uploaded documents
- Conversation memory for follow-up questions
- Adjustable retrieval depth
- View document sections used to generate answers
- Generate short and detailed summaries
- Extract important sections from documents
- Switch between multiple local LLMs
- Export chat sessions as Markdown
- Local processing through Ollama

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- PyPDFLoader

## How It Works

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Ollama Embeddings
    ↓
FAISS Vector Store
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Document Context
    ↓
Local LLM through Ollama
    ↓
Generated Answer with Sources