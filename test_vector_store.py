from embeddings.vector_store import load_and_chunk_text, create_faiss_index

chunks = load_and_chunk_text("data/website_text.txt")
print(f"Number of chunks: {len(chunks)}")

index, chunks, model = create_faiss_index(chunks)

query = "What does Python encourage in its community?"
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=2)

print("Top 2 chunks:")
for idx in I[0]:
    print(chunks[idx][:200], "...\n")