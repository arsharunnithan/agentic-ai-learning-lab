# Module 05 — AI Icebreaker Bot (LlamaIndex + Groq)

An AI-powered networking tool that reads a LinkedIn profile PDF 
and generates 5 personalized conversation icebreakers.

## What it does
- Upload any LinkedIn profile PDF
- LlamaIndex loads, chunks and embeds the profile
- Groq + Llama 3.1 generates personalized icebreakers
- Based on real career highlights, skills and achievements

## How to run
pip install -r requirements.txt
python icebreaker_bot.py

## Tech Stack
- Gradio — web interface
- LlamaIndex — document loading, chunking, vector index
- HuggingFace Embeddings — sentence-transformers/all-MiniLM-L6-v2
- FAISS — in-memory vector store
- Groq API — fast LLM inference
- Llama 3.1 8B — language model

## RAG Pipeline
LinkedIn PDF → SimpleDirectoryReader → SentenceSplitter (chunks)
→ HuggingFaceEmbedding → VectorStoreIndex → QueryEngine
→ Llama 3.1 → 5 Personalized Icebreakers

## Note
Add your Groq API key in icebreaker_bot.py before running.
Never commit real API keys to GitHub.

## Example Output
Given a LinkedIn profile, the bot generates questions like:
- "I noticed you worked with 2.5M+ grocery transactions — 
   what was the most surprising insight from that project?"
- "Your experience in econometric modeling is fascinating — 
   what's the most common misconception people have about it?"
