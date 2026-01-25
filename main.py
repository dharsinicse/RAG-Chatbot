from embeddings.vector_store import load_and_chunk_text, create_faiss_index
from llm.rag_chain import generate_answer
from utils.text_processing import clean_text

# context cleaning is now handled by utils.text_processing.clean_text

print("Loading data...")
chunks = load_and_chunk_text("data/website_text.txt")

print("Creating vector store...")
index, texts, model = create_faiss_index(chunks)

print("\n🤖 RAG Chatbot ready! (type 'exit' to quit)\n")

while True:
    question = input(">> ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    question = question.strip().lower()

    if not question:
        continue

    query_embedding = model.encode([question])
    D, I = index.search(query_embedding, k=4)

    # Relevance Guardrail
    if D[0][0] > 1.25:
        print("\nFinal Answer:\n")
        print("I apologize, but I cannot find that specific information in the current knowledge base. I can only answer questions related to the website's content.")
        print("-" * 60)
        continue

    raw_context = ""
    for score, idx in zip(D[0], I[0]):
        raw_context += texts[idx] + "\n"

    context = clean_text(raw_context)

    answer_gen = generate_answer(context, question)
    
    print("\nFinal Answer:\n")
    full_answer = ""
    for token in answer_gen:
        print(token, end="", flush=True)
        full_answer += token
    print("\n" + "-" * 60)
