# Module 03 — ChromaDB Similarity Search

Demonstrates how vector databases work by storing text documents 
as embeddings and performing semantic similarity search.

## What it does
- Stores 10 sample documents in ChromaDB as vector embeddings
- Takes a user query and finds the top 3 most similar documents
- Shows similarity percentage for each result
- Demonstrates semantic search vs keyword search

## How to run
pip install -r requirements.txt
python chroma_similarity.py

## Tech Stack
- Gradio — web interface
- ChromaDB — vector database (in-memory)
- SentenceTransformers — all-MiniLM-L6-v2 embedding model

## Key Concepts Learned
- Vector embeddings — converting text to numbers
- ChromaDB collections — like tables in a vector database
- Similarity search — finding meaning, not just keywords
- Distance vs similarity — how vectors are compared
- In-memory vs persistent ChromaDB

## How similarity search works
Documents → embeddings (384 numbers each) → stored in ChromaDB
Query → embedding → compared against all stored vectors
→ top 3 closest matches returned by distance

## Example
Query: "programming languages"
Result 1: "Python is the most popular language for AI" (57.8%)
Result 2: "Natural language processing helps computers..." (41.4%)
Result 3: "Data science combines statistics and programming..." (35.5%)

## Note
This uses in-memory ChromaDB — data resets on restart.
For persistent storage use: chromadb.PersistentClient(path="./db")
