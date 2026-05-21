# Module 04 — LangChain QA Bot

A document question-answering bot built with LangChain, FAISS, 
and Groq. Upload any PDF and ask questions — answers come only 
from your document.

## What it does

1. Upload a PDF
2. LangChain splits it into chunks
3. Chunks are embedded and stored in FAISS vector database
4. You ask a question
5. FAISS finds the most relevant chunks
6. Llama 3.1 answers based on those chunks only

## How to run

Install dependencies:
pip install -r requirements.txt

Run the app:
python qa_bot.py

## Architecture
PDF → Split into chunks → Embed → FAISS vector store
↓
User question → Embed → Similarity search → Top 3 chunks → LLM → Answer

## Tech Stack
- Gradio — web interface
- LangChain — RAG orchestration framework
- FAISS — local vector database
- HuggingFace Sentence Transformers — embedding model
- Groq + Llama 3.1 — LLM for answer generation

## Difference from Naive RAG (Module 03 App 5)

| Feature | Naive RAG | This QA Bot |
|---------|-----------|-------------|
| PDF handling | Dumps all text | Splits into chunks |
| Answer finding | Sends whole doc | Searches relevant chunks |
| Big PDF support | ❌ Token limit | ✅ Scales well |
| Vector database | ❌ | ✅ FAISS |
| Embeddings | ❌ | ✅ |

## Note
Add your Groq API key in qa_bot.py before running.
Never commit real API keys to GitHub.
