# Exploring Chroma DB

> A complete reference covering ChromaDB's architecture, deployment, filtering, HNSW indexing, and similarity search.

---

## Table of Contents

1. [What is Chroma DB?](#what-is-chroma-db)
2. [Chroma DB Capabilities](#chroma-db-capabilities)
3. [Deployment Options](#deployment-options)
4. [Architecture & Workflow](#architecture--workflow)
5. [Collections](#collections)
6. [Clients & Integrations](#clients--integrations)
7. [Metadata Filtering](#metadata-filtering)
8. [Document Filtering (Full-Text Search)](#document-filtering-full-text-search)
9. [Combining Filters](#combining-filters)
10. [Vector Indexes & HNSW](#vector-indexes--hnsw)
11. [Similarity Search in Chroma DB](#similarity-search-in-chroma-db)
12. [Performance Features](#performance-features)
13. [Use Cases](#use-cases)
14. [Summary](#summary)

---

## What is Chroma DB?

**Chroma DB** is a vector database specifically designed to support various retrieval tasks for AI applications. It combines vector search, full-text search, metadata filtering, and multi-modal retrieval in a single system.

> Chroma DB's core is written in **Rust**, delivering 3–5x speed improvements in querying and writing compared to a Python core.

---

## Chroma DB Capabilities

| Capability | Description |
|---|---|
| **Embedding storage** | Efficiently store and manage vector representations and their metadata |
| **Vector search** | Find semantically similar text using distance metrics (cosine, L2, dot product) |
| **Full-text search** | Find documents based on lexical/spelling similarity |
| **Document storage** | Store entire documents, not just their embeddings |
| **Metadata filtering** | Narrow search results based on document metadata |
| **Multi-modal retrieval** | Retrieve and manage images, audio, and text in a unified manner |

---

## Deployment Options

### Client-Server Mode (Default)
Chroma DB typically operates with a **client-server architecture**:
- The Chroma client connects to a Chroma server running in a **separate process**
- Server can be launched via the Chroma CLI or a **Docker image**
- Local or remote clients connect via the **HTTP protocol**

### Standalone Mode (Python only)
Both server and client functionalities run within a **single process**.
- Best for: quick feature testing or when the server always runs on the same machine as the client

---

## Architecture & Workflow

Chroma DB operates in multiple phases:

```
Phase 1 (Optional): Obtain Embeddings
    Convert text/images/data into vectors using an embedding model
    (OR let Chroma DB handle embedding automatically)
        ↓
Phase 2: Create Collections
    Collections = tables in a relational DB
        ↓
Phase 3: Store Data
    Add documents + embeddings (or let Chroma embed automatically)
        ↓
Phase 4: Collection Operations
    Delete, update, or rename collections
        ↓
Phase 5: Query & Retrieve
    Text or vector queries → results ranked by semantic similarity
    Filter by metadata or document content
```

> If you allow Chroma DB to handle embedding, it calculates and stores embeddings from documents **automatically in the background**.

### Typical Workflow Example

```python
import chromadb

client = chromadb.Client()

# Step 1: Create a collection
collection = client.create_collection(name="my_docs")

# Step 2: Add documents (Chroma handles embedding automatically)
collection.add(
    documents=["Document text here", "Another document"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
    ids=["id1", "id2"]
)

# Step 3: Query (Chroma embeds the query automatically)
results = collection.query(
    query_texts=["search query here"],
    n_results=3
)
```

> By default, Chroma uses **Euclidean (L2) distance** to identify the most similar chunks. It also supports **cosine distance** and **dot product**.

---

## Collections

Collections in Chroma DB are analogous to **tables in a relational database**. Each collection stores documents, their embeddings, and associated metadata.

```python
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="filter_demo",
    metadata={"description": "Demo collection"},
    configuration={"embedding_function": ef}
)
```

---

## Clients & Integrations

### Officially Supported Clients
- **Python** — maintained by ChromaCore team
- **JavaScript** — maintained by ChromaCore team

### Community-Supported Clients
Ruby · Java · Go · C# · Rust · PHP

### Framework Integrations
**LangChain** · **LlamaIndex** · **Ollama**

### Native Embedding Model Support
**Hugging Face** · **Google** · **OpenAI**

---

## Metadata Filtering

Metadata filtering uses the `where` parameter inside `.query()`, `.get()`, or `.delete()` methods.

### Basic Syntax

```python
# Simple equality match (two equivalent forms)
collection.get(where={"key": "value"})
collection.get(where={"key": {"$eq": "value"}})
```

### Comparison Operators

| Operator | Meaning | Types |
|---|---|---|
| `$eq` | Equal to | string, int, float |
| `$ne` | Not equal to | string, int, float |
| `$gt` | Greater than | int, float |
| `$gte` | Greater than or equal to | int, float |
| `$lt` | Less than | int, float |
| `$lte` | Less than or equal to | int, float |
| `$in` | Value is in list | list |
| `$nin` | Value is not in list | list |

### Logical Operators

```python
# AND — both conditions must be true
collection.get(
    where={
        "$and": [
            {"source": {"$eq": "langchain.com"}},
            {"version": {"$lt": 0.3}}
        ]
    }
)

# OR — either condition must be true
collection.get(
    where={
        "$or": [
            {"source": {"$eq": "langchain.com"}},
            {"source": {"$eq": "llamaindex.ai"}}
        ]
    }
)

# Using $in for list matching
collection.get(
    where={"source": {"$in": ["langchain.com", "llamaindex.ai"]}}
)
```

---

## Document Filtering (Full-Text Search)

Document filtering uses the `where_document` parameter to search within document **content** (not metadata).

> Also referred to as **full-text search** in Chroma DB. ⚠️ **Case-sensitive** — searching "Pandas" will not match "pandas".

```python
# Find documents containing a word
collection.get(where_document={"$contains": "pandas"})

# Find documents NOT containing a word
collection.get(where_document={"$not_contains": "library"})

# Combine with $or
collection.get(
    where_document={
        "$or": [
            {"$contains": "LangChain"},
            {"$contains": "Python"}
        ]
    }
)
```

---

## Combining Filters

Metadata and document filters can be used together in the same query:

```python
collection.get(
    where={"version": {"$gt": 0.1}},              # metadata filter
    where_document={
        "$or": [
            {"$contains": "LangChain"},            # document filter
            {"$contains": "Python"}
        ]
    }
)
```

### Filtering Within Similarity Search

```python
# Query with metadata filter
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={"topic": "animals"}
)

# Query with document content filter
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where_document={"$not_contains": "library"}
)

# Query with both filters combined
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={"topic": "animals"},
    where_document={"$not_contains": "library"}
)
```

> **Why filtering matters:** Without filters, a query for "polar bear" might incorrectly match a document about the *polars* Python library due to word overlap. Metadata or document filters can prevent such semantic mismatches.

---

## Vector Indexes & HNSW

### What is a Vector Index?

A **vector index** is a specialized data structure that organizes high-dimensional embeddings for fast similarity search — without needing to compare the query against every single vector (brute force).

Instead of a flat list, the index **clusters similar vectors together** or links them through proximity-based graphs, allowing the algorithm to skip large portions of the dataset early in the search.

### HNSW — Hierarchical Navigable Small World

**HNSW** is the sole indexing method used by Chroma DB. It is a graph-based algorithm for **Approximate Nearest Neighbor (ANN)** search.

**How it works:**

```
Upper layers  → Sparse overview of data — fast navigation
Lower layers  → Dense connections — detailed search
Bottom layer  → Contains ALL vectors

Search: Start at top → navigate toward query → descend layer by layer → return closest matches
```

Each vector connects to a few nearby neighbors forming a **"small world" network** — most vectors can be reached in just a few hops.

**Why HNSW?**

| Benefit | Description |
|---|---|
| **Fast** | Avoids scanning the entire dataset |
| **Accurate** | Delivers near-exact (approximate) results |
| **Scalable** | Handles millions to billions of vectors |
| **Versatile** | Works with L2, cosine, and dot product metrics |

### Configuring HNSW in Chroma DB

```python
collection = client.create_collection(
    name="my_collection",
    configuration={
        "hnsw": {
            "space": "cosine",        # distance metric: l2 | ip | cosine
            "ef_search": 100,         # candidate list size at query time
            "ef_construction": 100,   # candidate list size at index build time
            "max_neighbors": 16       # max connections per node
        },
        "embedding_function": ef
    }
)
```

### HNSW Parameters Explained

| Parameter | Default | Effect of Higher Value |
|---|---|---|
| `space` | `l2` | Sets distance metric — `l2`, `ip` (dot product), or `cosine` |
| `ef_search` | `100` | Better recall at query time — but slower queries |
| `ef_construction` | `100` | Better index quality — but slower build + more memory |
| `max_neighbors` | `16` | Denser graph — better search accuracy, more memory |

**Two categories of parameters:**
- **`ef_search`** → controls query-time breadth — the most direct lever for search quality vs. speed
- **`ef_construction` + `max_neighbors`** → control index build quality — affect accuracy but at the cost of longer build times and higher memory

---

## Similarity Search in Chroma DB

### Adding Documents

```python
collection.add(
    documents=[
        "Giant pandas are a bear species that lives in mountainous areas.",
        "A pandas DataFrame stores two-dimensional, tabular data",
        "I think everyone agrees that pandas are some of the cutest animals.",
        "A direct comparison between pandas and polars indicates polars is more efficient.",
    ],
    metadatas=[
        {"topic": "animals"},
        {"topic": "data analysis"},
        {"topic": "animals"},
        {"topic": "data analysis"},
    ],
    ids=["id1", "id2", "id3", "id4"]
)
```

### Querying

```python
results = collection.query(
    query_texts=["cats"],   # query passed as a list
    n_results=10            # number of results to return
)
```

- Results are **ranked by distance** — lowest distance = most similar
- `n_results` exceeding the collection size returns all documents
- Chroma DB **automatically embeds the query** — no manual embedding needed

### Interpreting Results

```python
# Output structure
{
    'ids': [['id3', 'id1', 'id2', 'id4']],
    'documents': [['...most similar...', '...', '...', '...least similar...']],
    'metadatas': [[{...}, {...}, {...}, {...}]],
    'distances': [[0.738, 0.835, 0.863, 0.929]]   # lower = more similar
}
```

> In the "cats" query example, the two animal-related pandas documents ranked highest — demonstrating that semantic search captures *contextual meaning*, not just keyword overlap.

---

## Performance Features

| Feature | Detail |
|---|---|
| **HNSW indexing** | Approximate nearest neighbor search — fast even at scale |
| **Rust core** | 3–5x speed improvement in querying and writing vs. Python |
| **Approximate search** | Trades marginal accuracy for massive speed gains |
| **Configurable trade-offs** | Tune `ef_search`, `ef_construction`, `max_neighbors` for your use case |

---

## Use Cases

| Use Case | How Chroma DB Helps |
|---|---|
| **Recommender systems** | Personalized suggestions based on user preference embeddings |
| **Document search engines** | Vector or full-text search across large document collections |
| **Image retrieval** | Find images using text queries via multi-modal retrieval |
| **AI chatbots / RAG** | Semantic search and retrieval for context augmentation |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Chroma DB** | Vector database optimized for retrieval tasks — vector search, full-text, metadata filtering, multi-modal |
| **Deployment** | Client-server (default) or standalone (Python only) |
| **Collections** | Equivalent to tables — store documents, embeddings, and metadata |
| **Default distance** | L2 (Euclidean) — also supports cosine and dot product |
| **Auto-embedding** | Chroma embeds documents and queries automatically if no embeddings are provided |
| **Metadata filter** | `where` parameter — operators: `$eq`, `$ne`, `$gt`, `$lt`, `$in`, `$nin`, `$and`, `$or` |
| **Document filter** | `where_document` parameter — `$contains`, `$not_contains` — case-sensitive |
| **Vector index** | Data structure enabling fast similarity search without brute-force comparison |
| **HNSW** | Multi-layered graph ANN algorithm — sole indexing method in Chroma DB |
| **`ef_search`** | Controls query-time recall vs. speed |
| **`ef_construction`** | Controls index build quality vs. build time and memory |
| **`max_neighbors`** | Controls graph density vs. memory usage |
| **Rust core** | 3–5x faster than Python-core alternatives |

---

*Notes based on: Exploring Chroma DB — Key Concepts, Filtering & HNSW Reading (Course 03 — Vector Databases for RAG)*
