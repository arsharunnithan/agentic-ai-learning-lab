# Course 03 — Module 04: ChromaDB Operations & Vector Databases in RAG

> A complete reference covering essential Chroma DB CRUD operations and how vector databases power the full RAG pipeline.

---

## Table of Contents

1. [Essential Chroma DB Operations](#essential-chroma-db-operations)
   - [Setup](#setup)
   - [Creating Collections](#creating-collections)
   - [Connecting to Existing Collections](#connecting-to-existing-collections)
   - [Modifying Collections](#modifying-collections)
   - [Adding Documents](#adding-documents)
   - [Getting Documents](#getting-documents)
   - [Updating Documents](#updating-documents)
   - [Deleting Documents](#deleting-documents)
   - [Setting the Distance Function](#setting-the-distance-function)
2. [How Vector Databases Power RAG](#how-vector-databases-power-rag)
   - [The Full RAG Pipeline](#the-full-rag-pipeline)
   - [What Vector Databases Handle in RAG](#what-vector-databases-handle-in-rag)
   - [Why Use a Vector Database for RAG?](#why-use-a-vector-database-for-rag)
   - [Common RAG Pitfalls](#common-rag-pitfalls)
   - [What Happens Outside the Vector Database](#what-happens-outside-the-vector-database)
3. [Summary](#summary)

---

## Essential Chroma DB Operations

### Setup

```python
import chromadb
from chromadb.utils import embedding_functions

# Define embedding model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create client
client = chromadb.Client()
```

---

### Creating Collections

Collections are the primary way to **organize data** in Chroma DB — analogous to tables in a relational database.

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "My first collection"},
    configuration={"embedding_function": ef}
)

print(collection.name)
# Output: my_collection
```

> Metadata can hold any key-value pairs useful for describing the collection's purpose and contents.

---

### Connecting to Existing Collections

```python
collection = client.get_collection(name="my_collection")

# Verify by checking metadata
print(collection.metadata)
# Output: {'description': 'My first collection'}
```

---

### Modifying Collections

Use the `modify` method to change a collection's **name or metadata**.

```python
collection.modify(
    name="my_collection_v2",
    metadata={"key": "value"}
)

print(collection.metadata)
# Output: {'key': 'value'}
```

> **Important:** Changes to the **embedding model** or **distance metric** cannot be made on an existing collection. You must clone the collection — which can be computationally expensive for large collections.

---

### Adding Documents

Use the `add` method to insert documents with optional metadata and required IDs.

```python
collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python"
    ],
    metadatas=[
        {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3}
    ],
    ids=["id1", "id2", "id3"]
)
```

- `documents` — list of text strings
- `metadatas` — list of dictionaries (no limitations on content)
- `ids` — **required** unique identifier for each document

> Chroma DB **automatically computes and stores embeddings** in the background.

---

### Getting Documents

Use the `get` method to retrieve documents from a collection.

```python
# Get all documents
result = collection.get()

# Get specific documents by ID
result = collection.get(ids=["id1", "id2"])

# Include embeddings in output (hidden by default)
result = collection.get(include=["embeddings"])
```

> Embeddings are stored but **not displayed by default** — pass `include=['embeddings']` to see them.

---

### Updating Documents

Use the `update` method to modify existing documents by their ID.

```python
collection.update(
    ids=["id1"],
    documents=["Updated document text about LangChain"],
    metadatas=[{"source": "langchain.com", "version": 0.9}]
)
```

> Chroma DB **automatically re-embeds** the updated document in the background immediately after the update is submitted.

---

### Deleting Documents

Delete by IDs, by metadata filter, or a combination of both.

```python
# Delete by specific IDs
collection.delete(ids=["id1", "id2"])

# Delete by metadata filter
collection.delete(where={"source": {"$eq": "llamaindex.ai"}})

# Combine IDs and filters
collection.delete(
    ids=["id1"],
    where={"version": {"$lt": 0.3}}
)
```

> Only documents for which the `where` condition resolves to `true` will be deleted.

---

### Setting the Distance Function

The distance function is configured via the HNSW `space` parameter **at collection creation time only**.

```python
collection = client.create_collection(
    name="my_collection",
    configuration={
        "hnsw": {
            "space": "cosine"   # options: l2 (default), cosine, ip
        },
        "embedding_function": ef
    }
)
```

| Value | Distance Function | Default |
|---|---|---|
| `l2` | Squared L2 (Euclidean) | Yes |
| `cosine` | Cosine distance | — |
| `ip` | Inner product (dot product) | — |

---

### Quick Reference — All Operations

| Operation | Method | Notes |
|---|---|---|
| Create collection | `client.create_collection()` | Set name, metadata, embedding function |
| Connect to collection | `client.get_collection()` | By name |
| Modify collection | `collection.modify()` | Name and metadata only |
| Add documents | `collection.add()` | IDs required; embedding automatic |
| Get documents | `collection.get()` | All or by IDs; embeddings hidden by default |
| Update documents | `collection.update()` | By ID; auto re-embeds |
| Delete documents | `collection.delete()` | By ID, filter, or both |

---

## How Vector Databases Power RAG

### The Full RAG Pipeline

```
Step 1: Gather source documents
        |
Step 2: Embed source documents / chunks
        |
Step 3: Store embeddings in vector database (e.g. Chroma DB)
        |
Step 4: Receive user prompt
        |
Step 5: Embed user prompt
        |
Step 6: Retriever selects best-matching chunks from vector store
        |
Step 7: Combine retrieved text + user prompt -> augmented prompt
        |
Step 8: Pass augmented prompt to LLM -> context-aware response
```

---

### What Vector Databases Handle in RAG

| RAG Step | Handled by Vector DB? |
|---|---|
| Embed source documents | Yes (automatic) |
| Store embeddings | Yes |
| Embed user prompt | Yes (automatic at query time) |
| Retrieve most relevant matches | Yes |
| Supply retrieved content for prompt augmentation | Yes |
| Chunking documents | No — done before data enters the DB |
| Advanced retrieval logic (re-ranking) | No — requires extra tools |
| Prompt augmentation | No — handled outside |
| LLM integration | No — not built into most vector DBs |

> Steps 2 and 5 (embedding) **can also be performed externally** — in that case, the vector database is used primarily for storing and retrieving pre-computed vectors.

---

### Why Use a Vector Database for RAG?

| Benefit | Detail |
|---|---|
| **Prevents critical mistakes** | Automatically uses the same embedding model for documents and queries — no risk of model mismatch |
| **Faster development** | Fewer moving parts, less custom logic, simpler and more maintainable codebase |
| **Better performance** | Built-in high-speed, scalable semantic search using advanced indexing (HNSW) — hard to match with custom-built alternatives |

---

### Common RAG Pitfalls

| Pitfall | Impact | Solution |
|---|---|---|
| **Using different embedding models** for documents and queries | Breaks retrieval entirely | Use the same model throughout; vector DBs handle this automatically |
| **Poor chunking strategy** | Chunks too large = irrelevant content; too small = loss of meaning | Choose chunk size that preserves meaning without excess noise |
| **Forgetting to re-embed after changes** | Stale embeddings that don't reflect new data, metrics, or models | Re-embed after any change; in Chroma DB this may require cloning the collection |
| **Not testing retrieval results** | Retrieved content may not be the best answer | Always evaluate and tune — small adjustments can significantly improve results |

---

### What Happens Outside the Vector Database

RAG frameworks like **LangChain** and **LlamaIndex** fill the gaps by wrapping around the vector database and managing the full pipeline:

| Task | Where It Happens |
|---|---|
| **Chunking** | Before data enters the vector DB |
| **Advanced retrieval** | Filtering, re-ranking — requires extra tools or framework logic |
| **Prompt augmentation** | Outside the DB — handled by the framework |
| **LLM integration** | Outside the DB — connected via LangChain or LlamaIndex |

> LangChain and LlamaIndex connect all the pieces — from document preparation to final LLM response — simplifying RAG application development end to end.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Collections** | Primary data organization unit in Chroma DB — like tables |
| **`create_collection`** | Creates a new collection with name, metadata, and embedding function |
| **`get_collection`** | Connects to an existing collection by name |
| **`modify`** | Changes name or metadata only — cannot modify embedding model or distance metric |
| **`add`** | Inserts documents with optional metadata and required IDs; auto-embeds |
| **`get`** | Retrieves documents; embeddings hidden by default |
| **`update`** | Modifies documents by ID; Chroma auto re-embeds |
| **`delete`** | Removes documents by ID, filter, or both |
| **`space` parameter** | Sets distance metric at collection creation — `l2` (default), `cosine`, `ip` |
| **RAG + vector DB** | Vector DB handles embedding, storage, retrieval, and prompt augmentation supply |
| **Why vector DB for RAG** | Prevents embedding mismatch, speeds development, delivers optimized search performance |
| **RAG pitfalls** | Embedding mismatch, bad chunking, stale embeddings, untested retrieval |
| **RAG frameworks** | LangChain / LlamaIndex wrap the vector DB to manage the full pipeline |

---

*Notes based on: Module 04 — Essential Chroma DB Operations + How Vector Databases Power RAG (Course 03 — Vector Databases for RAG)*
