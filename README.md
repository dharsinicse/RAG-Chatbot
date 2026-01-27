# 🤖 RAG-powered Website Chatbot
A high-performance Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and FLAN-T5. This assistant can ingest entire websites through recursive crawling, process the information using a persistent FAISS vector store that provides instant, accurate answers through a professional and minimalist interface.

## Problem Statement
Websites contain valuable information scattered across multiple pages, making it hard for users to quickly find accurate answers using traditional keyword search.

This project aims to:

1. Scrapes and processes website content

2. Retrieves the most relevant sections using semantic search

3. Generates answers strictly from the retrieved context

4. Clearly responds with “I don’t know based on the given information” when the answer is not present

## Strict Relevance & Guardrails
**Context Lock:** The AI stays strictly within your provided data—no assumptions, no outside knowledge. Every answer is grounded in the information you choose, so responses remain accurate, relevant, and fully under your control.

**Similarity Check:** Off-topic questions don’t slip through. A similarity check quietly evaluates each query, and anything outside scope is politely declined—keeping answers focused and trustworthy.

**Direct Answers:** Optimized to skip filler and fluff, responses that get straight to the facts, exactly when you need them.

## Key Features
**Instant Streaming Responses**

Experience ChatGPT-like interactions with word-by-word response streaming. No more waiting—answers appear as soon as the model begins generating.

**Smart Recursive Web Crawling**

Go beyond single-page scraping, the system explores an entire website - following internal links (up to depth 2) to build a comprehensive understanding of the target website content automatically and efficiently.

**Professional Neutral Interface**

A distraction-free aesthetic designed with a subtle dark-grey palette (#1e1e1e), glassmorphism containers, and modern typography (Outfit Google Font).

## Tech Stack
**Programming Language:** Python 3.13

**Web Scraping:** requests, BeautifulSoup

**Text Processing:** Recursive text chunking & cleaning utilities

**Embeddings:** sentence-transformers (semantic embeddings)

**Vector Database:** FAISS (IndexFlatL2) for efficient similarity search

**LLM:** Hugging Face Transformers (FLAN-T5 instruction-tuned model)

**RAG Framework:** Custom Retrieval-Augmented Generation pipeline

**UI:** Streamlit (interactive web-based chatbot interface)

## Project Structure

```
RAG-Chatbot/
├── .agent/
│   └── workflows/                 # Agent workflows / automation configs
│
├── app/
│   └── app.py                     # Streamlit UI for the RAG chatbot
│
├── embeddings/
│   ├── chunks.pkl                 # Serialized text chunks
│   ├── faiss_index.bin            # FAISS vector index
│   ├── vector_store.py            # Embedding & FAISS index logic
│   └── __pycache__/               # Python cache files
│
├── ingest/
│   ├── web_loader.py              # Website scraping (requests + BeautifulSoup)
│   └── __pycache__/               # Python cache files
│
├── llm/
│   ├── rag_chain.py               # RAG pipeline (retrieval + LLM response)
│   └── __pycache__/               # Python cache files
│
├── utils/
│   └── text_processing.py         # Text cleaning & chunking utilities
│
├── main.py                        # CLI-based chatbot
├── test_vector_store.py           # Vector store tests
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignored files
```

## How to Use
**1. Add Website**

In the sidebar, enter the website URL you want to query.

Click the “Add Website” button to load and process the content.

**2. Choose Crawl Depth**

Single Page: Scrapes only the given webpage.

Full Website: Crawls multiple pages within the same domain.

**3. Build Knowledge Base**

The system scrapes the website, splits the content into chunks,

Creates embeddings and stores them in a FAISS vector index.

This happens automatically after adding the website.

**4. Ask Questions (Chat Interface)**

Type your question in the chat input box.

Ask anything related to the website content.

Example:

“What is the purpose of this website?”

**5. Context-Aware Answers**

The chatbot retrieves the most relevant sections from the website.

If the answer is not found, it clearly responds:

“I don’t know based on the given information.”

**6. Chat History**

All previous questions and answers are saved in the sidebar.

This allows easy reference without re-asking questions.

Can be removed if needed.

## Clone the Repository
```
git clone - https://github.com/dharsinicse/RAG-Chatbot.git
cd RAG-Chatbot

install dependencies - pip install -r requirements.txt

run the application - streamlit run app/app.py
```
