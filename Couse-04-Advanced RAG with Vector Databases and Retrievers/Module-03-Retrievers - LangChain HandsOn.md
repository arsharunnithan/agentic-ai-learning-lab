# Advanced Retrievers with LangChain Hands-On

Hands-on comparison of 4 LangChain retrievers using a company 
HR and IT policy dataset.

## What it does
- Implements 4 different retrievers on the same dataset
- Compares results side by side via Gradio UI
- Demonstrates strengths and weaknesses of each retriever

## How to run
pip install -r requirements.txt
python retrievers.py

## Tech Stack
- Gradio — web interface with dropdown retriever selector
- LangChain Classic — retriever implementations
- FAISS — vector store for most retrievers
- ChromaDB — required for Self-Query Retriever
- HuggingFace Embeddings — sentence-transformers/all-MiniLM-L6-v2
- Groq API — LLM for Multi-Query and Self-Query retrievers

## The 4 Retrievers

### 1. Vector Store Retriever
- Basic similarity search
- Returns top-k most similar chunks
- Best for: simple direct questions
- Weakness: misses related concepts

### 2. Multi-Query Retriever
- LLM generates multiple query versions
- Searches with each version, combines unique results
- Best for: vague or complex questions
- Weakness: too many irrelevant results (noisy)

### 3. Self-Query Retriever
- LLM splits query into search + metadata filter
- Filters by department or topic automatically
- Best for: queries with implicit metadata filters
- Example: "show me only IT policies" → filters dept=IT

### 4. Parent Document Retriever
- Searches small child chunks for accuracy
- Returns larger parent chunks for context
- Best for: long documents needing more context
- Weakness: overkill for small documents

## Key Lessons Learned
- More results ≠ better results (Multi-Query)
- Self-Query needs specific metadata mentions in query
- Parent Document retriever shines on large real documents
- Each retriever has a sweet spot — choose based on use case

## Retriever Comparison
| Retriever | Results | Best for |
|-----------|---------|----------|
| Vector Store | 3 | Simple search |
| Multi-Query | 6+ | Better coverage |
| Self-Query | Filtered | Metadata filtering |
| Parent Document | Context-rich | Long documents |

## Note
Add your Groq API key in retrievers.py before running.
Never commit real API keys to GitHub.
