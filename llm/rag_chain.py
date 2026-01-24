from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

# Initialize model and tokenizer
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def generate_answer(context, question):
    prompt = f"""
[STRICT CONTEXT ONLY]
Task: Answer the Question using ONLY the information in the Context.
Rule 1: No greetings, no filler, no 'Based on the context'. Go straight to the answer.
Rule 2: If the info is missing, say: "I apologize, but I cannot find that specific information in the current knowledge base. I can only answer questions related to the website's content."
Rule 3: Keep it exactly 3 to 4 lines.

Context: 
{context}

Question: {question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=300, # Increased back up slightly for more room
        do_sample=True, # Allow a bit of variety to encourage detail
        top_k=50,
        top_p=0.95,
        temperature=0.7, # Slightly higher temperature to avoid repetitive brevity
        repetition_penalty=1.1 # Lowered slightly
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text