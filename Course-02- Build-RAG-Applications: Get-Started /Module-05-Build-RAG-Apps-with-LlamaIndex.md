# LlamaIndex — Document Ingestion, Chunking, Embeddings & Query Engines

> A complete reference covering LlamaIndex from document loading to query engines, plus a detailed comparison with LangChain.

---

## Table of Contents

1. [What is LlamaIndex?](#what-is-llamaindex)
2. [LlamaIndex in the RAG Pipeline](#llamaindex-in-the-rag-pipeline)
3. [Step 1 — Loading Documents](#step-1--loading-documents)
   - [The Document Class](#the-document-class)
   - [SimpleDirectoryReader](#simpledirectoryreader)
4. [Step 2 — Chunking (Nodes)](#step-2--chunking-nodes)
5. [Step 3 — Embedding & Vector Storage](#step-3--embedding--vector-storage)
6. [Step 4 — Retrieval](#step-4--retrieval)
7. [Step 5 — Response Generation](#step-5--response-generation)
   - [Response Synthesizer](#response-synthesizer)
   - [Query Engine](#query-engine)
8. [LangChain vs LlamaIndex](#langchain-vs-llamaindex)
9. [Summary](#summary)

---

## What is LlamaIndex?

**LlamaIndex** is a framework for building **LLM-powered context augmentation** — the process of making your data available to an LLM so it can perform tasks grounded in that data.

**Typical use cases:**

| Use Case | Description |
|---|---|
| **RAG (Question Answering)** | Retrieve relevant documents and generate grounded answers |
| **Chatbots** | Extend RAG with multi-turn conversation — follow-up questions, clarifications |
| **Document Understanding** | Read natural language and extract key details — names, dates, figures |
| **Data Extraction** | Semantically identify important information from structured or unstructured data |

---

## LlamaIndex in the RAG Pipeline

```
1. Load source documents
        ↓
2. Chunk documents into nodes
        ↓
3. Embed nodes → store vectors in VectorStoreIndex
        ↓
4. Receive user prompt → embed using same model
        ↓
5. Retriever fetches top-K similar nodes from vector store
        ↓
6. Augment prompt with retrieved nodes
        ↓
7. LLM generates context-aware response
```

---

## Step 1 — Loading Documents

### The Document Class

A LlamaIndex **Document** is a generic container for any source document. It holds:

| Component | Description |
|---|---|
| `id` | Unique identifier for the document |
| `embedding` | Placeholder for the document-level embedding (if embedding whole doc) |
| `metadata` | Dictionary for storing metadata (origin, date, author, etc.) |
| `relationships` | Dictionary linking the document to related documents |
| `text` | The actual text content of the document |

```python
from llama_index.core import Document

doc = Document(text="Hello LlamaIndex")
print(doc.dict())
```

**Supported formats:** `.txt` · `.pdf` · `.md` · `.csv` · `.json` · `.html` · Word documents · PowerPoint decks

---

### SimpleDirectoryReader

`SimpleDirectoryReader` is LlamaIndex's powerful built-in document loader. It handles many file types natively out of the box.

```python
from llama_index.core import SimpleDirectoryReader

# Load all files in a directory
documents = SimpleDirectoryReader("./data").load_data()

# Load recursively including subdirectories
documents = SimpleDirectoryReader("./data", recursive=True).load_data()

# Load specific files
documents = SimpleDirectoryReader(input_files=["./data/policy.pdf"]).load_data()

# Load specific file types only
documents = SimpleDirectoryReader("./data", required_exts=[".pdf", ".md"]).load_data()
```

> Output: a **list of LlamaIndex Document objects**

For additional connectors (SQL databases, RSS feeds, JSON, cloud storage), LlamaIndex provides **LlamaHub** — a registry of external data connectors.

---

## Step 2 — Chunking (Nodes)

In LlamaIndex, a **Node** is a text chunk. Chunking long documents into smaller pieces helps retain specific context per segment, leading to more precise embeddings and retrieval.

### SentenceSplitter

LlamaIndex's default and most commonly used chunker. It recursively splits on characters like newlines and periods.

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,      # max tokens per chunk
    chunk_overlap=50     # token overlap between consecutive chunks
)

nodes = splitter.get_nodes_from_documents(documents)
```

| Parameter | Description |
|---|---|
| `chunk_size` | Maximum number of tokens per chunk |
| `chunk_overlap` | Number of overlapping tokens between consecutive chunks — preserves context at boundaries |

> Output: a **list of LlamaIndex TextNode objects** — similar structure to Document objects

### Other Splitters

| Splitter | Description |
|---|---|
| **SentenceSplitter** | Default — recursive character-based, token-limited |
| **SemanticSplitterNodeParser** | Splits where sentence similarity falls below a threshold |
| **LangChainNodeParser** | Wrapper — use any LangChain splitter inside LlamaIndex |
| **HTMLNodeParser** | Splits HTML files by structure |
| **JSONNodeParser** | Splits JSON data |
| **MarkdownNodeParser** | Splits markdown by headers |

---

## Step 3 — Embedding & Vector Storage

LlamaIndex uses the **`VectorStoreIndex`** class to both generate embeddings and store them — all in one step.

### Simple In-Memory Usage

```python
from llama_index.core import VectorStoreIndex

# Embed nodes and store in memory (uses default embedding model)
index = VectorStoreIndex(nodes)
```

### Custom Embedding Model + Persistent Storage (ChromaDB)

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Define embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set up ChromaDB vector store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Embed and store
index = VectorStoreIndex(
    nodes,
    embed_model=embed_model,
    storage_context=storage_context
)
```

> `VectorStoreIndex` handles both embedding and storage — swap the backend without changing any downstream code.

---

## Step 4 — Retrieval

Create a retriever from the `VectorStoreIndex` and use it to fetch the most relevant nodes for a given prompt.

```python
# Default retriever
retriever = index.as_retriever()
results = retriever.retrieve("What is the company mobile policy?")

# Control how many results are returned (top-K)
retriever = index.as_retriever(similarity_top_k=5)
results = retriever.retrieve("What is the company mobile policy?")
```

- Results are returned as a **ranked list** — most similar nodes at the top
- `similarity_top_k` controls how many nodes are retrieved (default is typically 2)

---

## Step 5 — Response Generation

### Response Synthesizer

Combines **prompt augmentation + LLM querying + response generation** in one step. Takes the user's prompt and retrieved nodes as input.

```python
from llama_index.core import get_response_synthesizer

synthesizer = get_response_synthesizer()
response = synthesizer.synthesize("What is the mobile policy?", nodes=results)
print(response)
```

> Prompt embedding, augmentation, and LLM call all happen in the background automatically.

---

### Query Engine

The most powerful abstraction — combines **all RAG steps** into a single object:

```
Prompt embedding → Retrieval → Prompt augmentation → LLM querying → Response
```

```python
# Create query engine from the index
query_engine = index.as_query_engine()

# Run a query — everything happens automatically
response = query_engine.query("What is the company mobile policy?")
print(response)
```

**Customization options:**

```python
from llama_index.core import PromptTemplate

# Change the LLM
query_engine = index.as_query_engine(llm=your_custom_llm)

# Custom prompt template
template = "Context: {context_str}\nQuestion: {query_str}\nAnswer:"
qa_prompt = PromptTemplate(template)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

# Custom retriever with top-K
query_engine = index.as_query_engine(similarity_top_k=5)
```

**Comparison of response generation options:**

| Option | What it combines | Code complexity |
|---|---|---|
| **Manual** | Retriever + synthesizer called separately | More code, more control |
| **Response Synthesizer** | Prompt augmentation + LLM querying + generation | Moderate |
| **Query Engine** | Everything — embed, retrieve, augment, generate | Minimal — recommended |

---

## LangChain vs LlamaIndex

### Document Loading

| | LangChain | LlamaIndex |
|---|---|---|
| **Primary loader** | `DirectoryLoader` (uses `UnstructuredLoader` by default) | `SimpleDirectoryReader` (native, handles many formats) |
| **Specific loaders** | `TextLoader`, `CSVLoader`, `JSONLoader`, `WebBaseLoader`, `DoclingLoader` | Built into `SimpleDirectoryReader`; extras via LlamaHub |
| **Design philosophy** | Modular — relies heavily on external integrations | Native solutions first, external only when needed |

### Document Chunking

| | LangChain | LlamaIndex |
|---|---|---|
| **Default splitter** | `RecursiveCharacterTextSplitter` | `SentenceSplitter` |
| **Character-based** | `CharacterTextSplitter` | — |
| **Token-based** | `TokenTextSplitter` | `SentenceSplitter` (token-based by default) |
| **Semantic** | `SemanticChunker` | `SemanticSplitterNodeParser` |
| **File-structured** | `MarkdownHeaderTextSplitter`, HTML, JSON, code | HTML, JSON, Markdown, code node parsers |
| **Cross-framework** | — | `LangChainNodeParser` — use any LangChain splitter in LlamaIndex |

### Embedding & Vector Storage

| | LangChain | LlamaIndex |
|---|---|---|
| **Embedding + storage** | Two separate steps | One command: `VectorStoreIndex(nodes)` |
| **In-memory store** | `InMemoryVectorStore` | Default behavior of `VectorStoreIndex` |
| **External DBs** | Chroma, FAISS, Milvus, PGVector (via integrations) | Chroma, FAISS, Milvus (wrapped by `VectorStoreIndex`) |
| **Metadata handling** | Manual setup; varies by backend | Automatic — stored in `VectorStoreIndex` |
| **Backend flexibility** | More granular control per store | Backend-agnostic — swap store without changing downstream code |

### Prompt Augmentation & LLM Response

| | LangChain | LlamaIndex |
|---|---|---|
| **Prompt augmentation** | Separate step — easy to customize | Combined with LLM step in synthesizer/query engine |
| **LLM call** | Manual — `llm.invoke(messages)` | Automatic — handled by synthesizer or query engine |
| **Abstraction level** | Lower — more manual steps, more control | Higher — query engine handles everything |
| **Template customization** | Easy — augmentation is isolated | Slightly harder — combined with LLM step |

### Overall Comparison

| Dimension | LangChain | LlamaIndex |
|---|---|---|
| **Design** | Modular, integration-heavy | Native solutions, batteries-included |
| **Ease of use** | More setup required | Simpler defaults, less boilerplate |
| **Customization** | Easier — steps are isolated | Harder — steps are combined |
| **Flexibility** | Higher — granular control | Lower — but covers most typical cases |
| **Best for** | Complex, custom RAG pipelines | Rapid development, standard RAG workflows |

> Both frameworks are capable of handling most typical RAG workflows. Choose **LangChain** for maximum flexibility and granular control. Choose **LlamaIndex** for faster development with sensible defaults.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **LlamaIndex** | Framework for LLM-powered context augmentation — RAG, chatbots, document understanding |
| **Document class** | Container with ID, text, metadata, relationships, and embedding placeholder |
| **SimpleDirectoryReader** | Powerful native loader — handles many formats, directories, subdirectories, and file type filters |
| **Node** | LlamaIndex term for a text chunk |
| **SentenceSplitter** | Default chunker — recursive, token-based, with `chunk_size` and `chunk_overlap` |
| **VectorStoreIndex** | Embeds and stores chunks in one command — works in-memory or with persistent DBs |
| **Retriever** | Created via `index.as_retriever()` — fetches top-K similar nodes |
| **Response Synthesizer** | Combines prompt augmentation + LLM querying + response generation |
| **Query Engine** | All-in-one — embed, retrieve, augment, generate with a single `.query()` call |
| **LlamaIndex vs LangChain** | LlamaIndex = simpler defaults; LangChain = more flexible and modular |

---

*Notes based on: LlamaIndex Document Ingestion & Chunking + Vector Stores to Query Engines + LangChain vs LlamaIndex Reading (Course 02)*
