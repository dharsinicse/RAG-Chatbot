# 🤖 RAG-powered Website Chatbot
A high-performance Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and FLAN-T5. This assistant can ingest entire websites through recursive crawling, process the information using a persistent FAISS vector store and provide instant, accurate answers through a professional, minimalist interface.

## Key Features
**Instant Streaming Responses**
Experience ChatGPT-like interactions with word-by-word response streaming. No more waiting—answers appear as soon as the model begins generating.

**Smart Recursive Web Crawling**
Don't just scrape a single page—ingest a whole domain. The bot can automatically follow internal links (up to depth 2) to build a comprehensive knowledge base about your target website.

**Professional Neutral Interface**
A distraction-free aesthetic designed with a sleek dark-grey palette (#1e1e1e), glassmorphism containers, and modern typography (Outfit Google Font).

## Strict Relevance & Guardrails
Context Lock: The AI is strictly forbidden from using outside general knowledge. It only answers using your provided data.
Similarity Check: Off-topic questions are caught by a similarity threshold and politely declined to ensure accuracy.
Direct Answers: Optimized to skip greetings and filler, going straight to the facts.

## Tech Stack
Frontend: Streamlit (Custom CSS)
AI Model: Google FLAN-T5 Large
Vector Store: FAISS (with Persistence)
Embeddings: Sentence Transformers (all-mpnet-base-v2)
Scraping: Requests & BeautifulSoup4

## How to Use
Add Website: Click the "Add Website" button in the sidebar.
Choose Depth: Select "Single Page" or "Full Website" to crawl the domain.
Chat: Once the knowledge base is updated, ask anything about the website!
History: Your chats are saved in the sidebar for easy reference.

## Project Structure
text
├── app/
│   └── app.py
├── embeddings/
│   └── vector_store.py
├── ingest/
│   └── web_loader.py
├── llm/
│   └── rag_chain.py
├── utils/
│   └── text_processing.py
└── requirements.txt
