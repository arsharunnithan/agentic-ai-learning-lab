# YouTube QA Bot

An AI-powered tool that extracts transcripts from YouTube videos
and answers questions about the video content using RAG.

## What it does
- Paste any YouTube URL with captions/subtitles
- Automatically extracts the full transcript
- Splits transcript into chunks and stores in FAISS
- Ask any question about the video content
- Groq + Llama 3.1 answers based on transcript only

## How to run
pip install -r requirements.txt
streamlit run youtube_qa.py

## Tech Stack
- Streamlit — web interface (new framework!)
- LangChain — RAG pipeline
- YoutubeLoader — transcript extraction
- FAISS — vector store
- HuggingFace Embeddings — sentence-transformers/all-MiniLM-L6-v2
- Groq API — fast LLM inference
- Llama 3.1 — language model

## How it works
YouTube URL
↓
YoutubeLoader extracts transcript
↓
RecursiveCharacterTextSplitter → chunks
↓
HuggingFaceEmbeddings → vectors
↓
FAISS vector store
↓
User asks question → retriever finds top 3 chunks
↓
Groq + Llama 3.1 answers from transcript

## Key Concepts Learned
- YoutubeLoader for transcript extraction
- Streamlit vs Gradio — different UI frameworks
- @st.cache_resource — caches models so they don't reload every time
- Real RAG on long content (1647 chunks from one video!)

## Streamlit vs Gradio
| | Gradio | Streamlit |
|--|--------|-----------|
| Port | 7860 | 8501 |
| Run command | python app.py | streamlit run app.py |
| Best for | AI demos | Data apps |
| Layout | Predefined | More flexible |

## Note
Add your Groq API key in youtube_qa.py before running.
Never commit real API keys to GitHub.
