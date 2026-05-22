# Advanced Retrievers for RAG

> A complete reference covering LangChain and LlamaIndex retrievers — from basic vector store retrieval to advanced fusion strategies.

---

## Table of Contents

1. [What is a LangChain Retriever?](#what-is-a-langchain-retriever)
2. [LangChain Retrievers](#langchain-retrievers)
   - [Vector Store-Based Retriever](#vector-store-based-retriever)
   - [Multi-Query Retriever](#multi-query-retriever)
   - [Self-Query Retriever](#self-query-retriever)
   - [Parent Document Retriever](#parent-document-retriever)
3. [LlamaIndex Index Types](#llamaindex-index-types)
4. [LlamaIndex Retrievers](#llamaindex-retrievers)
   - [Vector Index Retriever](#vector-index-retriever)
   - [BM25 Retriever](#bm25-retriever)
   - [Document Summary Index Retriever](#document-summary-index-retriever)
   - [Auto Merging Retriever](#auto-merging-retriever)
   - [Recursive Retriever](#recursive-retriever)
   - [Query Fusion Retriever](#query-fusion-retriever)
5. [Fusion Strategies](#fusion-strategies)
6. [Retriever Recommendations by Use Case](#retriever-recommendations-by-use-case)
7. [Summary](#summary)

---

## What is a LangChain Retriever?

A **LangChain retriever** is an interface that:
- Accepts a **string query** as input
- Returns a **list of documents or chunks** as output

It is more general than a vector store — its purpose is to **retrieve** documents, not necessarily store them.

```
Query (string) → Retriever → List of relevant documents/chunks
```

---

## LangChain Retrievers

### Vector Store-Based Retriever

The **simplest retriever** — retrieves documents directly from a vector database.

**How it works:**
```
Load source documents
    ↓
Split into chunks
    ↓
Embed chunks → store in vector store
    ↓
Retriever plugs into the vector store
    ↓
Query arrives → embed query → compare with stored chunks
    ↓
Return most similar chunks
```

**Two search methods:**

| Method | Description |
|---|---|
| **Similarity Search** | Returns chunks most similar to the query by embedding distance |
| **Maximum Marginal Relevance (MMR)** | Balances relevance AND diversity — avoids returning redundant chunks |

> **MMR** selects documents that are both highly relevant to the query AND minimally similar to already selected documents — ensuring broader coverage of different aspects of the query.

```python
# Similarity search retriever
retriever = vectorstore.as_retriever(search_type="similarity")

# MMR retriever
retriever = vectorstore.as_retriever(search_type="mmr")

# Invoke
docs = retriever.invoke("email policy")
```

Does **not** require an LLM — pure vector comparison.

---

### Multi-Query Retriever

Overcomes the limitation that subtle wording changes can produce different results. Uses an **LLM to generate multiple versions** of the query, then takes the **unique union** of results across all queries.

**Why it helps:**
- Different phrasings surface different relevant documents
- Compensates when embeddings don't fully capture the semantics

```python
from langchain.retrievers import MultiQueryRetriever

# Base retriever (any retriever works here)
base_retriever = vectorstore.as_retriever()

# Multi-query retriever uses LLM to generate query variants
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm   # e.g. Mixtral 8x7B
)

docs = multi_query_retriever.invoke("original query")
```

**Flow:**
```
Original query
    ↓
LLM generates multiple query variants
    ↓
Each variant retrieves a set of documents
    ↓
Unique union of all results returned
```

---

### Self-Query Retriever

Handles documents that have **both text and metadata**. Converts the query into two components:

1. **Semantic string** — what to look up in the vector store
2. **Metadata filter** — structured filter to apply alongside the search

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Describe the metadata fields to help the LLM build filters
metadata_field_info = [
    AttributeInfo(name="year", description="Year of release", type="integer"),
    AttributeInfo(name="director", description="Director name", type="string"),
    AttributeInfo(name="rating", description="IMDB rating", type="float")
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief description of movies",
    metadata_field_info=metadata_field_info
)

# Natural language query with implicit metadata filter
docs = retriever.invoke("I want to watch a movie rated higher than 8.5")
# Returns only movies where rating > 8.5
```

> Standard retrievers only see document text — the self-query retriever is the only one that can leverage **metadata for filtering**.

---

### Parent Document Retriever

Solves the tension between two conflicting chunking goals:
- **Small chunks** → accurate embeddings
- **Large chunks** → sufficient context for useful responses

**Solution:** Embed small chunks for retrieval, but return the larger parent chunks.

```python
from langchain.retrievers import ParentDocumentRetriever

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,           # stores small chunk embeddings
    docstore=InMemoryStore(),          # stores large parent chunks
    child_splitter=child_splitter,     # splits into small chunks for embedding
    parent_splitter=parent_splitter    # splits into large chunks to return
)

retriever.add_documents(documents)
docs = retriever.invoke("smoking policy")
# Returns large parent chunks, not the small child chunks
```

**Flow:**
```
Small chunks embedded → stored in vector store
    ↓
Query arrives → small chunks retrieved by similarity
    ↓
Parent IDs looked up
    ↓
Large parent chunks returned
```

---

## LlamaIndex Index Types

LlamaIndex provides three core index types, each suited for different retrieval strategies:

| Index Type | How It Works | Best For |
|---|---|---|
| **VectorStoreIndex** | Stores vector embeddings per chunk | Semantic search, general RAG pipelines |
| **DocumentSummaryIndex** | Generates and stores document summaries at index time | Large, diverse document sets that exceed LLM context windows |
| **KeywordTableIndex** | Extracts keywords and maps them to content chunks | Exact keyword matching, hybrid or rule-based search |

---

## LlamaIndex Retrievers

### Vector Index Retriever

Uses **vector embeddings** to find semantically relevant content.

- Best for: general-purpose search and RAG pipelines
- Works with `VectorStoreIndex`

```python
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("your query here")
```

---

### BM25 Retriever

A **keyword-based** retrieval method — finds documents by exact keyword match rather than semantic similarity.

**Background — TF-IDF (the foundation of BM25):**

| Component | What it measures |
|---|---|
| **TF (Term Frequency)** | How often a word appears in a document |
| **IDF (Inverse Document Frequency)** | How rare that word is across all documents |
| **TF-IDF score** | TF × IDF — highlights words frequent in one doc but rare overall |

**BM25 improvements over TF-IDF:**
- **Term frequency saturation** — reduces the impact of repeatedly occurring terms
- **Document length normalization** — adjusts scores for document length

```python
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
results = bm25_retriever.retrieve("exact keyword query")
```

---

### Document Summary Index Retriever

Uses **document summaries** (not raw documents) to identify relevant content — then returns the **original full documents**.

**Two versions:**

| Version | Method | Trade-off |
|---|---|---|
| **LLM-based** | LLM determines most relevant summaries | More accurate, but slower and more expensive |
| **Embedding-based** | Semantic similarity between query and summary embedding | More efficient for large collections |

> Regardless of version, the **original documents** are always returned — not the summaries.

---

### Auto Merging Retriever

Designed for **long documents** — preserves broader context by working with a hierarchical chunk structure.

**How it works:**
- Documents split into **parent and child nodes** (hierarchical chunking)
- Small child chunks are retrieved first
- If **enough child nodes from the same parent** are retrieved → the parent node is returned instead

```
Query → retrieve child chunks → enough from same parent? → return parent node
```

This consolidates related content and preserves broader context.

---

### Recursive Retriever

Designed for documents with **cross-references** between nodes — follows links from one node to another.

- Supports **chunk references** (e.g. a section referencing another section)
- Supports **metadata references** (e.g. citations in academic papers)
- Retrieves related content **across documents or layers of abstraction**

---

### Query Fusion Retriever

Combines results from **multiple retrievers** (e.g. vector-based + keyword-based) and optionally generates **multiple query variations** using an LLM.

```python
from llama_index.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rerank",
    num_queries=4   # LLM generates 4 query variants
)
```

---

## Fusion Strategies

The Query Fusion Retriever supports three strategies for merging results:

| Strategy | How It Works | Best For |
|---|---|---|
| **Reciprocal Rank Fusion** | Assigns higher scores to documents appearing near the top of any ranked list — doesn't rely on score magnitudes | General use — robust and reliable |
| **Relative Score Fusion** | Normalizes scores within each result set by dividing by the maximum score — preserves relative confidence | When retriever confidence levels matter |
| **Distribution-Based Fusion** | Uses statistical normalization (z-score or percentile ranking) to combine results | Handling variability between retrievers with different score scales |

---

## Retriever Recommendations by Use Case

| Use Case | Recommended Retriever(s) | Reason |
|---|---|---|
| **General Q&A** | Vector Index + BM25 (fused) | Combines semantic relevance with keyword matching |
| **Technical documents** | BM25 primary + Vector Index secondary | Prioritizes exact terms while adding contextual flexibility |
| **Long documents** | Auto Merging Retriever | Returns larger parent chunks when enough child chunks match |
| **Research papers** | Recursive Retriever | Follows citations and metadata links across documents |
| **Large document sets** | Document Summary Index Retriever → then Vector Search | Narrows candidates first, then retrieves most pertinent content |

---

## Summary

| Retriever | Framework | Key Mechanism |
|---|---|---|
| **Vector Store-Based** | LangChain | Embedding similarity or MMR against a vector store |
| **Multi-Query** | LangChain | LLM generates query variants → unique union of results |
| **Self-Query** | LangChain | Splits query into semantic search + metadata filter |
| **Parent Document** | LangChain | Small chunks for embedding, large parent chunks returned |
| **Vector Index** | LlamaIndex | Semantic search via embeddings — general purpose |
| **BM25** | LlamaIndex | Keyword-based, exact match with TF-IDF improvements |
| **Document Summary Index** | LlamaIndex | Uses summaries to filter, returns original documents |
| **Auto Merging** | LlamaIndex | Hierarchical chunks — returns parent if enough children match |
| **Recursive** | LlamaIndex | Follows cross-references and metadata links between nodes |
| **Query Fusion** | LlamaIndex | Combines multiple retrievers using fusion ranking strategies |
| **MMR** | LangChain | Balances relevance AND diversity — reduces redundancy |
| **Reciprocal Rank Fusion** | LlamaIndex | Scores by rank position — robust across retrievers |
| **Relative Score Fusion** | LlamaIndex | Normalizes scores — preserves retriever confidence |
| **Distribution-Based Fusion** | LlamaIndex | Statistical normalization — handles score variability |

---

*Notes based on: Advanced Retrievers for RAG — LangChain Parts 1 & 2 + LlamaIndex Advanced Retrievers (Course 04)*
