    
## ChromaDB Similarity Search

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


## Exercise 01 — Employee Similarity Search
**File:** `employee_search.py`
- Stores 5 employee records in ChromaDB as vector embeddings
- Converts structured dictionaries to text descriptions
- Performs semantic similarity search on any query
- Returns top 3 most relevant employees with match percentage

## Key Concepts Learned
- Converting structured data to text for embedding
- ChromaDB collection creation with embedding function
- List comprehensions in Python
- Semantic search vs keyword search
- Distance to similarity conversion: (1 - distance) * 100
  
## Note
This uses in-memory ChromaDB — data resets on restart.
For persistent storage use: chromadb.PersistentClient(path="./db")


## Exercise 02: AI Food Recommendation System

An advanced food recommendation system with three distinct 
approaches to similarity search and conversational AI.

## What it does
- Basic semantic similarity search on food descriptions
- Advanced filtered search by cuisine and calories
- RAG chatbot for intelligent food recommendations

## How to run
pip install -r requirements.txt
python food_recommender.py

## Tech Stack
- Gradio — web interface with tabs
- ChromaDB — persistent vector database
- SentenceTransformers — all-MiniLM-L6-v2 embedding model
- Groq API — fast LLM inference
- Llama 3.1 — language model for chatbot

## Three Features

### Tab 1 — Basic Search
Type any craving and get top 3 similar foods using 
semantic similarity search.

### Tab 2 — Filtered Search
Search with hard filters:
- Filter by cuisine (Indian, Italian, Mexican etc)
- Filter by maximum calories
ChromaDB first filters, then searches remaining foods.

### Tab 3 — RAG Food Chatbot
Conversational AI that:
- Retrieves relevant foods from ChromaDB
- Passes them as context to Llama 3.1
- Generates personalized recommendations in natural language

## Key Concepts Learned
- Persistent ChromaDB (data survives restart)
- get_or_create_collection vs create_collection
- Metadata filtering ($eq, $lte operators)
- gr.Tab() for multi-tab Gradio interfaces
- NLP with SentenceTransformers for embeddings
- Real RAG limitation — false positives in similarity search

## Key Difference from Exercise 01
Exercise 01 — in-memory ChromaDB (resets on restart)
Exercise 02 — persistent ChromaDB (saved to food_db/ folder)


## Note
Add your Groq API key in food_recommender.py before running.
Never commit real API keys to GitHub.

