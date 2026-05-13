# Retrieval-Augmented Generation (RAG) — Deep Dive

> A detailed reference covering the RAG process, embeddings, retrieval mechanics, and implementation details.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Why RAG? — LLM Limitations](#why-rag--llm-limitations)
3. [RAG vs Long Context Models](#rag-vs-long-context-models)
4. [The RAG Process — Step by Step](#the-rag-process--step-by-step)
5. [Text Embedding in Depth](#text-embedding-in-depth)
   - [Prompt Encoding](#prompt-encoding)
   - [Knowledge Base Encoding](#knowledge-base-encoding)
6. [Retrieval — Finding Relevant Context](#retrieval--finding-relevant-context)
7. [Augmented Prompt Creation](#augmented-prompt-creation)
8. [RAG Implementation Details](#rag-implementation-details)
9. [Summary](#summary)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a machine learning technique that integrates **information retrieval** with **generative AI** to produce accurate, context-aware responses.

RAG equips LLMs with access to **external data sources** — enabling them to answer questions that go beyond their training data without requiring the model to be retrained.

**Two core components:**

| Component | Role |
|---|---|
| **Retriever** | The core of RAG — finds and fetches relevant information from the knowledge base |
| **Generator** | The LLM — produces a natural language response using the retrieved context |

---

## Why RAG? — LLM Limitations

### Without RAG

An LLM without RAG is limited to:
- Information provided by the user in the prompt
- Information learned during training

This leads to three common failure modes:

| Problem | Description |
|---|---|
| **Inaccurate responses** | Especially for domain-specific or specialized queries |
| **Outdated information** | Training data has a cutoff; new developments are unknown |
| **Hallucinations** | Model fabricates believable but incorrect answers |

Additionally, responses **lack verifiable sources**, making it impossible to trace or confirm accuracy.

### With RAG

```
User Prompt
    ↓
Retriever fetches relevant content from external data store
    ↓
Augmented Prompt (original prompt + retrieved text)
    ↓
LLM generates a grounded, sourced response
```

RAG addresses all three failure modes:
- **Accuracy** — grounded in retrieved primary source data
- **Currency** — update the data store, not the model
- **Hallucination** — less reliance on training memory; can say "I don't know" when the data store can't answer

---

## RAG vs Long Context Models

Modern LLMs can handle context lengths of **128,000 tokens or more** (~96,000 English words). This raises the question: why use RAG at all?

### Limitations of Long Context Without RAG

| Limitation | Detail |
|---|---|
| **Input dependency** | Users must already have and provide the source information |
| **Limited capacity** | 128K tokens still can't fit very long texts (e.g. War and Peace exceeds 560,000 words) |
| **Redundancy** | Irrelevant or repeated content dilutes LLM focus — the "needle in a haystack" problem |
| **Processing time** | More tokens = longer response times |
| **Cost** | More tokens = higher computational and financial cost |

### How RAG Addresses Each

| Limitation | RAG Solution |
|---|---|
| Input dependency | Connects to an external data store users don't need to provide |
| Limited capacity | Retrieves only the most relevant chunks — fits within context limits |
| Redundancy | Only pertinent information is passed to the LLM |
| Processing time | Shorter augmented prompts = faster responses |
| Cost | Fewer tokens = lower generation costs |

---

## The RAG Process — Step by Step

```
1. Gather Sources
        ↓
2. Embed Sources (chunk → embed → store in vector DB)
        ↓
3. Store Vectors in vector database
        ↓
4. Receive User Prompt
        ↓
5. Embed User Prompt (same embedding model)
        ↓
6. Retrieve Relevant Data (compare vectors, fetch top-K matches)
        ↓
7. Create Augmented Prompt (retrieved text + original prompt)
        ↓
8. Generate Response (LLM produces final answer)
```

### Step Details

| Step | What Happens |
|---|---|
| **Gather Sources** | Collect documents, policies, reports. Preprocess into machine-friendly formats (e.g. PDF → plain text) |
| **Embed Sources** | Chunk documents → run through embedding model → produces fixed-length numeric vectors |
| **Store Vectors** | Store vectors in a vector database (ChromaDB, FAISS, Milvus) with a chunk ID as the key |
| **User Prompt** | Can be standalone or include prior conversation history (managed by LangChain/LlamaIndex memory tools) |
| **Embed Prompt** | Same embedding model as sources — ensures vector compatibility for comparison |
| **Retrieve** | Compare prompt vector to stored vectors using distance metrics; select top-K most similar chunks |
| **Augmented Prompt** | Combine retrieved text + original prompt via concatenation or structured template |
| **Generate** | LLM processes augmented prompt → produces a grounded, sourced response |

---

## Text Embedding in Depth

Embedding is the process of converting text into **high-dimensional numeric vectors** that capture semantic meaning.

### How Embedding Works

```
Text
  ↓ Tokenization
Tokens (words, sub-words, punctuation)
  ↓ Token ID Assignment
Unique numerical IDs (consistent for same token)
  ↓ Neural Network Processing
Fixed-length vector (captures semantic meaning)
```

### Prompt Encoding

When encoding a user's prompt:

1. **Token Embedding** — each token is converted into a high-dimensional vector using a pre-trained model (e.g. BERT, GPT)
2. **Vector Averaging** — all token vectors are averaged into a **single vector representation** for the entire prompt

> The averaged vector captures the overall meaning of the prompt in a compact, comparable format.

### Knowledge Base Encoding

Large documents cannot be inserted directly — they must be **chunked** first:

1. **Chunking** — break the document into smaller, manageable pieces of text
2. **Embedding** — each chunk is converted into a vector using a pre-trained token embedding model
3. **Averaging** — token vectors within each chunk are averaged into one vector per chunk
4. **Indexing** — chunk vectors are stored in a vector database with a **chunk ID** as the key

**Example — Company Mobile Policy:**
```
Full Policy Document (too large to insert directly)
    ↓ Chunked into 7 text segments
    ↓ Each chunk embedded into a vector
    ↓ Stored in vector DB with IDs: 0, 1, 2, 3, 4, 5, 6
```

---

## Retrieval — Finding Relevant Context

Once the prompt and knowledge base are both vectorized, the system **compares them** to find the most relevant chunks.

### Distance Metrics

| Metric | What it Measures | Best For |
|---|---|---|
| **Dot Product** | Direction AND magnitude — prioritizes overall alignment | When vector magnitude matters |
| **Cosine Similarity** | Direction only — measures angular difference | When only semantic direction matters |

### Top-K Selection

The system selects the **top K** chunks whose vectors are closest to the prompt vector.

- **K is a hyperparameter** — typically 3–5 chunks are retrieved
- Example: from a 7-chunk policy document, chunks with IDs `0, 2, 6` may be selected as most relevant

### Retrieval Strategies

| Strategy | Description |
|---|---|
| Most relevant chunk only | Single best-matching text segment |
| Full document retrieval | Fetch the entire document containing the relevant chunk |
| Multiple document retrieval | Fetch several relevant documents for broader context |

---

## Augmented Prompt Creation

The retrieved text is combined with the original user prompt to form the **augmented prompt**.

**Two common methods:**

| Method | Description |
|---|---|
| **Simple concatenation** | Retrieved text is appended directly to the user's prompt |
| **Structured template** | A prompt template with defined sections for user input, retrieved text, and LLM instructions |

**The augmented prompt structure:**
```
[Instruction: use the retrieved context to answer the question]
[Retrieved Context: relevant chunks from knowledge base]
[User Question: original prompt]
```

The LLM then processes this combined input and generates a response grounded in the retrieved source data. Responses can be further refined using **predefined output templates** for consistent presentation.

---

## RAG Implementation Details

### Embedding Models vs LLMs

| | Embedding Model | LLM |
|---|---|---|
| **Purpose** | Converts text to vectors | Generates natural language responses |
| **Examples** | BERT, sentence-transformers | GPT, LLaMA, IBM Granite |
| **Used for** | Encoding sources and prompts | Generating final answer |
| **Must match?** | ✅ Same model for sources and prompts | N/A |

### Vector Databases

| Database | Notes |
|---|---|
| **ChromaDB** | Lightweight, easy to set up locally |
| **FAISS** | Facebook AI's library, fast for large datasets |
| **Milvus** | Enterprise-scale, feature-rich |

### RAG Knowledge Base Sources

| Source Type | Examples |
|---|---|
| Internal documents | Company policies, HR documents, manuals |
| Open sources | Internet, public APIs |
| Large documents | Research papers, legal contracts, reports |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **RAG** | Combines retrieval from external sources with LLM generation |
| **Two components** | Retriever (finds context) + Generator (produces response) |
| **Chunking** | Large documents split into smaller pieces for efficient retrieval |
| **Embedding** | Text → tokens → token IDs → neural network → fixed-length vector |
| **Vector averaging** | All token vectors in a chunk/prompt averaged into one representative vector |
| **Distance metrics** | Dot product (magnitude + direction) vs Cosine (direction only) |
| **Top-K retrieval** | K nearest vectors selected from the knowledge base (K is a hyperparameter) |
| **Augmented prompt** | Retrieved context + original user prompt combined before feeding to LLM |
| **vs Long context** | RAG is more efficient, cheaper, and handles larger data stores |
| **Vector databases** | ChromaDB, FAISS, Milvus — specialized storage for embedding vectors |

---

*Notes based on: Retrieval-Augmented Generation (RAG) — Video Lecture + Reading (Course 02)*
