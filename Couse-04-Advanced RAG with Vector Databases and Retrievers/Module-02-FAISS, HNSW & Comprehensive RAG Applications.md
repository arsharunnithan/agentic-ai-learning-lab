# Module 02 — FAISS, HNSW & Comprehensive RAG Applications

> A complete reference covering FAISS vs Chroma DB, all FAISS index types, and a deep dive into the HNSW algorithm.

---

## Table of Contents

1. [FAISS Overview](#faiss-overview)
2. [FAISS vs Chroma DB vs Milvus](#faiss-vs-chroma-db-vs-milvus)
3. [FAISS Index Types](#faiss-index-types)
   - [Flat Index](#flat-index)
   - [Inverted File Index (IVF)](#inverted-file-index-ivf)
   - [Locality-Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)
   - [Hierarchical Navigable Small World (HNSW)](#hierarchical-navigable-small-world-hnsw)
4. [Extending FAISS with Milvus](#extending-faiss-with-milvus)
5. [When to Use Which Tool](#when-to-use-which-tool)
6. [HNSW Deep Dive](#hnsw-deep-dive)
   - [The Building Blocks](#the-building-blocks)
   - [How HNSW Search Works](#how-hnsw-search-works)
   - [How the HNSW Index is Built](#how-the-hnsw-index-is-built)
   - [Key Parameters](#key-parameters)
   - [Trade-offs and Limitations](#trade-offs-and-limitations)
7. [Summary](#summary)

---

## FAISS Overview

**FAISS** (Facebook AI Similarity Search) is a **library** built by Meta for fast vector search on a single machine.

| Property | Detail |
|---|---|
| **Type** | Library (not a database) |
| **Hardware** | CPU or GPU — single machine |
| **Interface** | Code only — no built-in server or UI |
| **Strength** | Full control and high performance |
| **Weakness** | No metadata support, no distributed scaling natively |

---

## FAISS vs Chroma DB vs Milvus

| Feature | FAISS | Chroma DB | Milvus |
|---|---|---|---|
| **Type** | Library | Full database | Full database |
| **Deployment** | Single-node only | Single-node or server | Distributed |
| **Index types** | Many (Flat, IVF, LSH, HNSW, etc.) | HNSW only | Multiple (uses FAISS internally) |
| **Metadata support** | No | Yes — store and filter | Yes — hybrid queries |
| **Scaling** | No native distributed scaling | Limited | Production-grade distributed |
| **LangChain / LlamaIndex** | Yes | Yes | Yes |
| **Best for** | High-performance local search | Quick AI prototyping | Large-scale production |

---

## FAISS Index Types

Each index type makes different trade-offs between **speed, memory, and accuracy**.

### Flat Index

A **brute-force** approach — compares the query embedding against every single vector in the store.

- **Distance metric:** Euclidean (L2) or dot product
- **Returns:** Top-K nearest vectors, ordered closest to farthest
- **Accuracy:** Exact — 100%
- **Speed:** Slowest — does not scale well to large datasets

> Use Flat Index when you need exact results and your dataset is small.

---

### Inverted File Index (IVF)

Speeds up search by **clustering vectors** using k-means into **Voronoi cells** around centroids.

**How it works:**
```
All vectors clustered into cells (k-means)
    |
Query vector arrives
    |
Search limited to vectors in the nearest cells only
    |
Top-K results returned
```

- **Accuracy:** Slightly lower — nearby vectors may fall in different cells
- **Speed:** Much faster than Flat for large datasets
- **Memory:** Moderate

> Use IVF when you need faster search and can tolerate a small accuracy trade-off.

---

### Locality-Sensitive Hashing (LSH)

Uses **hash functions** that map similar vectors to the same hash bucket, enabling fast and memory-efficient search.

**How it works:**
```
Vectors hashed into buckets (similar vectors → same bucket)
    |
Query vector hashed
    |
Search limited to vectors in matching buckets
```

- **Best for:** High-dimensional sparse data (e.g. text embeddings)
- **Speed:** Fast
- **Memory:** Efficient
- **Accuracy:** Neither the fastest nor the most accurate — a middle-ground option

---

### Hierarchical Navigable Small World (HNSW)

Organizes vectors into a **hierarchy of graph layers** for both fast and accurate approximate search. See the full deep dive section below.

- **Top layers:** Sparse — act like express highways for fast navigation
- **Bottom layer:** Dense — contains all vectors for detailed local search
- **Search:** Starts at top, descends layer by layer, refining at each level
- **Best for:** Large datasets requiring both speed and accuracy

---

## Extending FAISS with Milvus

**Milvus** is a full vector database that uses **FAISS as one of its core indexing engines** and adds the missing capabilities:

| Gap in FAISS | Milvus Solution |
|---|---|
| No metadata support | Stores and filters metadata alongside vectors |
| No distributed scaling | Full distributed deployment support |
| No hybrid queries | Supports queries like "Find similar items under $50" |

> **Example hybrid query:** "Find vectors similar to X where `price < 50` and `category = 'electronics'`"

---

## When to Use Which Tool

| Scenario | Tool |
|---|---|
| Full control and performance on a single machine | **FAISS** |
| Quick AI development, prototyping, metadata-rich queries | **Chroma DB** |
| Scalable, production-ready, distributed, hybrid search | **Milvus** |

---

## HNSW Deep Dive

### The Building Blocks

#### 1. Small World Networks

Small-world networks have two key properties:
- **High clustering coefficient** — nodes form tight-knit groups
- **Low average path length** — any node can be reached in just a few steps

> Real-world analogy: Any two people in the world are connected by ~6 degrees of separation. HNSW applies this to data — any vector can reach any other in a few hops.

#### 2. Navigable Networks — Greedy Routing

In a navigable network, connections guide the search in the right direction.

The search process:
```
Start at entry point
    |
Look at all directly connected neighbors
    |
Move to the neighbor closest to the target
    |
Repeat until no closer neighbor exists
```

Time complexity: **O(log^k n)** — significantly faster than brute-force O(n).

#### 3. Hierarchical Structure — The Layers

HNSW creates **multiple graph layers** inspired by skip lists:

| Layer | Characteristics |
|---|---|
| **Top layer** | Very few nodes — long-distance "highway" connections |
| **Middle layers** | More nodes — medium-range connections |
| **Bottom layer (Layer 0)** | ALL nodes — short-range, fine-grained connections |

Node layer assignment uses an **exponentially decaying probability distribution** — most nodes only appear at Layer 0, very few reach the top.

---

### How HNSW Search Works

```
Step 1: Enter the graph at a random point in the top layer
        |
Step 2: Greedy search in the current layer
        - Compute distance between entry point and query
        - Compute distances to all neighbors in current layer
        - Move to the closest neighbor
        - Repeat until no closer neighbor exists
        |
Step 3: Move down one layer using the best candidate as entry point
        |
Step 4: Repeat greedy search in new layer
        |
Step 5: Continue descending until reaching Layer 0
        |
Step 6: Final greedy search in Layer 0
        |
Step 7: Return approximate nearest neighbor(s)
```

**Why it's efficient:** In a 12-node example, HNSW needs only ~8 distance computations vs. 12 for brute force. In real-world large datasets, the efficiency gains are dramatically larger.

---

### How the HNSW Index is Built

**Step 1:** Start with an empty graph — the first inserted point becomes the entry point.

**Step 2:** Assign a layer height to each new node using exponential probability:
- Most nodes → Layer 0 only
- Fewer nodes → Layer 1
- Even fewer → Layer 2, and so on

**Step 3:** Insert the node:
```
a. Start greedy search from the top layer
b. Move to the best candidate found in each layer
c. At each layer (top to bottom): connect the new node to its M closest neighbors
d. Connections are bidirectional
```

**Step 4:** Repeat for every new data point — over time this builds the multi-layered graph.

---

### Key Parameters

| Parameter | Controls | Effect of Higher Value |
|---|---|---|
| **M** | Max connections per node | Better recall + more memory usage |
| **efConstruction** | Search breadth during index build | Better graph quality + slower build + more memory |
| **efSearch** | Search breadth during querying | Better accuracy + slower query |
| **ml** (level multiplier) | Probability of appearing in higher layers | Controls hierarchy shape |

**Tuning guidelines:**
- Start with defaults: `M=16`, `efConstruction=200`
- Increase `M` for higher recall (up to M=64)
- Adjust `efSearch` at query time to tune speed vs. accuracy
- Use benchmarking to find optimal settings for your dataset

**Two categories:**
- **`efSearch`** → query-time lever — most direct control over speed vs. accuracy
- **`M` + `efConstruction`** → build-time levers — affect graph quality but require rebuilding to change

---

### Trade-offs and Limitations

| Limitation | Detail |
|---|---|
| **Approximate results** | Typical recall: 90–99% — may miss exact nearest neighbor |
| **Parameter tuning required** | M, efConstruction, efSearch all need tuning for optimal performance |
| **Best for static datasets** | Frequent insertions/deletions degrade graph quality over time — may require periodic rebuilding |
| **Memory usage** | Higher M and efConstruction increase memory consumption significantly |

**Supported distance metrics:** Works best with L2 (Euclidean) and cosine similarity. Other metrics may require modifications.

**When NOT to use HNSW:**
- You need guaranteed exact results → use Flat Index
- Dataset is very small → traditional methods are simpler
- Memory is critically constrained
- Data changes very frequently

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **FAISS** | Library for fast local vector search — full control, no metadata, no distributed scaling |
| **Chroma DB** | Full vector database — easier setup, HNSW only, metadata support, great for prototyping |
| **Milvus** | Production-grade database — uses FAISS internally, adds metadata + distributed scaling |
| **Flat Index** | Brute-force exact search — accurate but slow at scale |
| **IVF Index** | Clusters vectors into cells — faster but slight accuracy trade-off |
| **LSH** | Hash-based search — memory-efficient, good for sparse high-dimensional data |
| **HNSW** | Multi-layer graph — best balance of speed and accuracy for large datasets |
| **Small world network** | Any node reachable in few hops — forms the foundation of HNSW |
| **Greedy routing** | Move to the closest neighbor at each step — O(log^k n) complexity |
| **Layer structure** | Top = sparse highways; Bottom = dense local streets |
| **M** | Max connections per node — higher = better recall + more memory |
| **efConstruction** | Build-time search breadth — higher = better index quality |
| **efSearch** | Query-time search breadth — primary speed vs. accuracy lever |
| **HNSW recall** | Typically 90–99% — near-exact results with far less computation |
| **HNSW limitation** | Best for static datasets — degrades with frequent updates |

---

*Notes based on: Module 02 — FAISS vs Chroma DB + HNSW Deep Dive Reading (Course 04 — Comprehensive RAG Applications)*
