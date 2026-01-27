import sys
import os
import streamlit as st
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.vector_store import (
    load_and_chunk_text, 
    create_faiss_index, 
    save_index, 
    load_existing_index
)
from llm.rag_chain import generate_answer
from ingest.web_loader import load_website_text
from utils.text_processing import clean_text

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp { 
        background-color: #1e1e1e;
        color: #e2e8f0; 
    }
    
    header[data-testid="stHeader"] { background-color: transparent !important; }
    section[data-testid="stSidebar"] { 
        background-color: #262626 !important; 
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
    }
    
    /* Glassmorphism Containers */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(5px);
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stTextInput > div > div > input { 
        background-color: rgba(255, 255, 255, 0.05) !important; 
        color: white !important; 
        border-radius: 12px !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 1rem !important;
    }
    
    /* Welcome Cards */
    .stButton button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        height: auto !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
    }
    .stButton button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: #94a3b8 !important;
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        color: #fff !important;
    }
    
    /* Typography Overrides */
    h1 { font-weight: 700 !important; letter-spacing: -1px !important; color: #fff; }
    h3 { font-weight: 600 !important; color: #cbd5e1; }
    
    /* Sidebar Polish */
    [data-testid="stSidebarNav"] { padding-top: 2rem !important; }
    
    .sidebar-divider { 
        height: 1px; 
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%); 
        margin: 2rem 0; 
    }
    
    .kb-label { 
        display: flex; 
        align-items: center; 
        gap: 10px; 
        margin-bottom: 0.5rem;
    }

    /* Custom Add Website Button */
    div.stButton > button[key="add_web_btn"], .add-web-btn {
        background: transparent !important;
        border: 1px dashed rgba(148, 163, 184, 0.4) !important;
        color: #94a3b8 !important;
        text-align: center !important;
        padding: 0.8rem !important;
        border-radius: 12px !important;
    }
    div.stButton > button[key="add_web_btn"]:hover {
        border-color: #cbd5e1 !important;
        color: white !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    /* Dialog Box */
    div[data-role="dialog"] {
        background-color: #1a1a1a !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 20px !important;
    }

    /* Profile Display */
    .profile-row { display: flex; align-items: center; padding: 10px 0; gap: 12px; }
    .profile-avatar { 
        width: 32px; 
        height: 32px; 
        background: #475569 !important; 
        color: white !important; 
        border-radius: 8px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-weight: 700;
        font-size: 0.9rem;
    }
    .profile-name { font-weight: 600; color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "website_text.txt")

@st.cache_resource
def initialize_system():
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings", "faiss_index.bin")
    chunks_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings", "chunks.pkl")
    
    # Ensure directories exist for deployment
    data_dir = os.path.dirname(DATA_PATH)
    embeddings_dir = os.path.dirname(index_path)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(embeddings_dir): os.makedirs(embeddings_dir)
    
    index, chunks = load_existing_index(index_path, chunks_path)
    model = SentenceTransformer("all-mpnet-base-v2")
    
    if index and chunks:
        return index, chunks, model
        
    if not os.path.exists(DATA_PATH): return None, None, None
    chunks = load_and_chunk_text(DATA_PATH)
    index, texts, model = create_faiss_index(chunks)
    
    save_index(index, texts, index_path, chunks_path)
    
    return index, texts, model

from sentence_transformers import SentenceTransformer
index, texts, model = initialize_system()

def save_chat_to_history():
    """Archives the current chat session."""
    if st.session_state.messages:
        first_user_msg = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "New Chat")
        title = (first_user_msg[:25] + '...') if len(first_user_msg) > 25 else first_user_msg
        
        st.session_state.chat_history.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "messages": list(st.session_state.messages)
        })

@st.dialog("Add Website")
def add_website_dialog():
    url = st.text_input("Enter Website URL", placeholder="https://example.com")
    depth_choice = st.selectbox("Crawl Depth", ["Single Page", "Full Website (Depth 2)"])
    max_depth = 1 if depth_choice == "Single Page" else 2
    
    if st.button("Scrape & Add"):
        if not url:
            st.warning("Please enter a URL")
            return
            
        with st.status("Gathering Information...") as status:
            status.write(f"Scraping started on: {url}...")
            text = load_website_text(url, max_depth=max_depth)
            
            if text:
                status.write("Updating Knowledge Base...")
                try:
                    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings", "faiss_index.bin")
                    chunks_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings", "chunks.pkl")
                    if os.path.exists(index_path): os.remove(index_path)
                    if os.path.exists(chunks_path): os.remove(chunks_path)
                    
                    with open(DATA_PATH, "w", encoding="utf-8") as f:
                        f.write(text)
                    st.cache_resource.clear()
                    status.update(label="Knowledge Base Updated!", state="complete")
                    st.success("Website indexed! Reloading...")
                    st.rerun()
                except Exception as e:
                    status.update(label="Failed to update", state="error")
                    st.error(f"Failed to save data: {e}")
            else:
                status.update(label="Extraction failed", state="error")
                st.error("Failed to extract text.")

with st.sidebar:
    st.markdown("### 🤖 RAG-Chatbot")
    
    if st.button("New Chat", use_container_width=True):
        save_chat_to_history()
        st.session_state.messages = []
        st.rerun()

    search_query = st.text_input("Search chats", placeholder="Search...", label_visibility="collapsed")
    
    st.caption("RECENT ACTIVITY")
    
    if st.session_state.chat_history:
        if st.button("Clear History", key="clear_history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if search_query:
        filtered_history = [c for c in st.session_state.chat_history if search_query.lower() in c['title'].lower()]
    else:
        filtered_history = st.session_state.chat_history

    if not filtered_history:
        st.markdown("<div style='color:#666;font-size:0.8rem;font-style:italic;'>No matching chats</div>", unsafe_allow_html=True)
    else:
        for chat in reversed(filtered_history):
            if st.button(f"{chat['title']}", key=chat['id'], use_container_width=True):
                st.session_state.messages = list(chat["messages"])
                st.rerun()

    st.markdown("<div style='height: 35vh;'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="kb-label">
            <span style="flex-grow:1;">Knowledge Base ({index.ntotal if index else 0})</span>
            <span style="font-size: 0.8rem; color: #888;">▼</span>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Add Website", key="add_web_btn", help="Scrape a new URL"):
        add_website_dialog()

    st.markdown("""
        <div class="sidebar-divider"></div>
        <div class="profile-row">
            <div class="profile-avatar">D</div>
            <div class="profile-name">Dharsini</div>
        </div>
    """, unsafe_allow_html=True)


if not index:
    st.info("👋 Welcome! Your knowledge base is currently empty.")
    st.markdown("To get started, please add a website URL so the chatbot can learn about its content.")
    if st.button("Add your first website", use_container_width=True):
        add_website_dialog()
    st.stop()

def process_input(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    query_emb = model.encode([prompt])
    D, I = index.search(query_emb, k=4)
    
    threshold = 1.25
    if D[0][0] > threshold:
        answer = "I apologize, but I cannot find that specific information in the current knowledge base. I can only answer questions related to the website's content."
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return

    retrieved = [texts[i] for i in I[0] if i < len(texts)]
    # Use double newlines to separate chunks clearly for the model
    context = clean_text("\n\n".join(retrieved))
    
    # If cleaning removed everything but we had a good match, use raw
    if not context.strip():
        context = retrieved[0]
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        stream = generate_answer(context, prompt)
        full_response = st.write_stream(stream)
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response
    })

# Main Area Container
main_container = st.container()

if len(st.session_state.messages) == 0:
    with main_container:
        st.markdown("<h1 style='text-align:center;margin-top:10vh;font-size:3.5rem;'>How can I help you today?</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#94a3b8;font-size:1.2rem;margin-bottom:5vh;'>Ask anything about the website given</p>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1])
        sug = [
            ("What is this?", "What is the main purpose and topic of this website?"),
            ("Summarize", "Provide a comprehensive summary of the website content."),
            ("Features", "List the key features, products, or services mentioned."),
            ("Contact", "What are the contact details, emails, or support channels provided?")
        ]
        
        with c1:
            if st.button(sug[0][0], use_container_width=True): 
                main_container.empty()
                process_input(sug[0][1])
            if st.button(sug[2][0], use_container_width=True): 
                main_container.empty()
                process_input(sug[2][1])
            
        with c2:
            if st.button(sug[1][0], use_container_width=True): 
                main_container.empty()
                process_input(sug[1][1])
            if st.button(sug[3][0], use_container_width=True): 
                main_container.empty()
                process_input(sug[3][1])
else:
    with main_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    # If first message, clear the welcome screen by not rendering it (happens on rerun)
    # But for the current run, we can just call process_input
    # To avoid the flicker, we should check if messages will be empty AFTER this block
    if len(st.session_state.messages) == 0:
        main_container.empty() # Clear the welcome screen immediately
    
    process_input(prompt)
