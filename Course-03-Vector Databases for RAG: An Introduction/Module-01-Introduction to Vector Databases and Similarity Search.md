# Course 03 — Vector Databases for RAG: An Introduction
## Module 01 — Introduction to Vector Databases

> A complete reference covering vector database concepts, types, comparisons with traditional databases, and real-world applications.

---

## Table of Contents

1. [What is a Vector Database?](#what-is-a-vector-database)
2. [What is a Vector?](#what-is-a-vector)
3. [Vector Databases vs Traditional Databases](#vector-databases-vs-traditional-databases)
4. [Vector Libraries vs Vector Databases](#vector-libraries-vs-vector-databases)
5. [Types of Vector Databases](#types-of-vector-databases)
6. [Dedicated Vector Databases vs Databases with Vector Search](#dedicated-vector-databases-vs-databases-with-vector-search)
7. [Real-World Applications](#real-world-applications)
8. [Similarity Search & Distance Metrics](#similarity-search--distance-metrics)
9. [Summary](#summary)

---

## What is a Vector Database?

A **vector database** is a specialized database designed to store, index, and query **high-dimensional vector data** rapidly. Instead of organizing data in tables like traditional databases, vector databases represent data as vectors in a **multi-dimensional space**.

**Core capabilities:**
- **Similarity search** — find items closest to a query vector
- **Nearest neighbor queries** — locate the most similar items
- **Distance calculations** — measure relationships between vectors
- **Clustering and classification** — group and categorize items

**Why vector databases matter for AI/ML:**
- Natural way to store and explore machine learning data
- Easily integrate into ML pipelines
- Speed up the creation and release of AI-powered applications
- Essential component of RAG systems

---

## What is a Vector?

A **vector** is an array of numerical values where each number represents a specific **feature or attribute** of the data. Each numerical value is a **dimension**.

**Example — Book Representation:**

| Book | Genre | Pages | Year | Rating |
|---|---|---|---|---|
| Fiction | 1 | 350 | 2003 | 4.5 |
| Non-Fiction | 2 | 250 | 2015 | 4.8 |
| Science Fiction | 3 | 400 | 1990 | 4.2 |

Each book = a vector: `[genre, pages, year, rating]`

A similarity search for *"science fiction, ~200 pages, rating 4.7–5.0"* compares your query vector against all stored vectors and returns the closest matches — without scanning the entire database.

**Data types vectors can represent:**
Images · Sounds · Text · Pattern data · Map data · Genomic information · Sensor data

**Techniques used:**
- **Distributed computing** — handles big datasets
- **Indexing** — enables fast lookups
- **Parallel processing** — speeds up query execution

---

## Vector Databases vs Traditional Databases

| Function | Traditional (Relational) DB | Vector DB |
|---|---|---|
| **Data representation** | Tables, rows, and columns — structured data | Multi-dimensional vectors — unstructured/complex data |
| **Query language** | SQL (SELECT, INSERT, UPDATE, DELETE) | Similarity search, nearest neighbor queries |
| **Indexing** | B-trees — optimized for exact lookups | Metric trees, hashing — optimized for high-dimensional spaces |
| **Scalability** | Challenging — requires sharding or resource augmentation | Designed for horizontal scaling via distributed architectures |
| **Best for** | Business applications, transactional systems, structured data | Scientific research, NLP, multimedia analysis, AI/ML pipelines |
| **Search type** | Exact match | Approximate / similarity-based |

### How Each Stores Data

**Relational Database:**
```
Table: Books
| ID | Title        | Genre     | Pages |
|----|------------- |-----------|-------|
| 1  | Dune         | Sci-Fi    | 412   |
| 2  | Sapiens      | Non-Fic   | 443   |
```
Connected via primary and foreign keys. Queried with SQL.

**Vector Database:**
```
Each item → embedding vector → stored in multi-dimensional space
Image → [0.23, 0.87, 0.11, ...]
Text  → [0.54, 0.02, 0.73, ...]
```
Queried by distance/similarity between vectors.

---

## Vector Libraries vs Vector Databases

| Feature | Vector Library | Vector Database |
|---|---|---|
| **Storage** | In-memory only | In-memory or persistent |
| **CRUD support** | Read and update only | Full CRUD (Create, Read, Update, Delete) |
| **Production use** | Lightweight, development-focused | Enterprise-level production deployments |
| **Algorithms** | Pre-configured similarity algorithms | Pre-configured + customizable indexing |
| **Example** | FAISS (as a library) | Milvus, ChromaDB, Pinecone |

---

## Types of Vector Databases

### 1. In-Memory Vector Databases
Store vectors directly in RAM for **ultra-fast read/write** operations.

- **Best for:** Real-time analytics, recommendation systems, low-latency applications
- **Examples:** RedisAI, TorchServe

### 2. Disk-Based Vector Databases
Store vectors on disk with sophisticated indexing and compression.

- **Best for:** Large datasets that exceed memory capacity
- **Examples:** Annoy, Milvus, ScaNN

> **Annoy** (Approximate Nearest Neighbors Oh Yeah) — stores vectors on disk and constructs indexes for fast approximate nearest neighbor searches.

### 3. Distributed Vector Databases
Spread vector data across **multiple nodes or servers** for horizontal scalability and fault tolerance.

- **Best for:** Massive datasets, high throughput workloads
- **Examples:** FAISS, Elasticsearch with Vector Plugin, Dask-ML

> **FAISS** (Facebook AI Similarity Search) — partitions data across nodes for scalable, speedy similarity searches in high-dimensional spaces.

### 4. Graph-Based Vector Databases
Model data as graphs where **nodes and edges** represent vector attributes or embeddings.

- **Best for:** Relationship analysis, knowledge graphs, social network analysis
- **Examples:** Neo4j, Amazon Neptune, TigerGraph

> **Neo4j** — stores vectors as node properties; supports social network analysis, recommendation systems, and knowledge graphs.

### 5. Time-Series Vector Databases
Manage and analyze **data collected over time** represented as vectors.

- **Best for:** Temporal pattern analysis, anomaly detection, IoT monitoring
- **Examples:** InfluxDB, TimescaleDB, Prometheus

> **InfluxDB** — stores vectors alongside time-stamped data; enables trend forecasting and system metric monitoring.

---

## Dedicated Vector Databases vs Databases with Vector Search

### Dedicated Vector Databases

Purpose-built for storing, indexing, querying, and analyzing large volumes of vector data.

**Key characteristics:**
- Use specialized data structures: **inverted indexes**, **product quantization**, **locality-sensitive hashing (LSH)**
- Natively support vector operations: nearest neighbor search, similarity search, distance calculations
- Designed for horizontal scalability across distributed systems
- Speed-optimized for high-dimensional data
- Customizable indexing and search parameters

**Popular dedicated vector databases:**

| Database | Notes |
|---|---|
| **FAISS** | Facebook AI Similarity Search — distributed, optimized for high-dimensional spaces |
| **Annoy** | Approximate nearest neighbors — fast, disk-based |
| **Milvus** | Enterprise-grade, disk-based, feature-rich |
| **ChromaDB** | Lightweight, popular in LangChain/LlamaIndex workflows |
| **Pinecone** | Managed cloud vector database |

---

### Databases with Vector Search Support

Regular database systems or data processing frameworks that add vector search capability via **extensions or plugins** — not built primarily for vector operations.

**Storage formats:** BLOBs, arrays, or user-defined types (UDTs)

**Notable vendors:**

| Vendor | Vector Support |
|---|---|
| **SingleStore** | Vector processing — integrates with IBM watsonx.ai |
| **Elasticsearch** | Vector add-on plugin |
| **PostgreSQL** | PostGIS extension for spatial vectors |
| **MySQL** | Built-in vector search indexes |
| **RedisAI** | In-memory vector functions |
| **MongoDB** | Vector search with flexible schemas |
| **Apache Cassandra** | Vector search support |

> These databases offer flexibility but may not match the speed and optimization of dedicated vector databases. Evaluate based on your performance, scalability, and functionality requirements.

---

## Real-World Applications

### Image & Video Analysis

| Capability | Description |
|---|---|
| **Feature extraction** | Store high-dimensional feature vectors (color histograms, texture, deep learning embeddings) |
| **Similarity search** | Locate visually similar images or video segments |
| **Real-time processing** | Video surveillance, object recognition, live event analysis |

> **Example:** A photo-sharing app stores embeddings of user photos. When a new image is added, the app compares its embedding to existing ones and suggests similar photos for tagging or album organization.

---

### Recommendation Systems

| Capability | Description |
|---|---|
| **Embedding storage** | Store numerical representations of items/entities |
| **Nearest neighbor search** | Find the most similar items to what a user interacted with |
| **Cross-domain suggestions** | Combine embeddings across categories for richer recommendations |
| **Scalability** | Handle millions of concurrent users with fast query processing |

> **Example:** A streaming service stores movie embeddings. After you watch a film, the system retrieves embeddings of related movies from the vector database and recommends what to watch next.

---

### Geospatial Analysis & Location Services

| Capability | Description |
|---|---|
| **Spatial indexing** | R-tree or quadtree indexing for GPS locations, polygons, addresses |
| **Spatial queries** | Proximity searches, range queries, spatial joins |
| **Location-based suggestions** | Nearby events, services, places of interest |
| **Real-time routing** | Fleet management, dynamic vehicle routing, traffic hotspot detection |

> **Example:** A navigation app stores GPS locations of restaurants as vectors. When you search nearby, it queries the vector database and returns results within your specified distance.

---

### Social Media & Marketing

| Capability | Description |
|---|---|
| **Distributed storage** | Horizontal scaling across nodes for big data processing |
| **User profile management** | Store and query user interests, behaviors, and preferences |
| **Trend analysis** | Fast delivery of trending topics to influencers and advertisers |
| **Autoscaling** | Dynamically adjust hardware and cloud resources to handle demand |

> **Example:** A social platform stores user profiles with interests like cycling, running, and swimming. As users grow, the database autoscales hardware to maintain performance while tracking clicks and preferences.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Vector database** | Specialized DB storing data as high-dimensional vectors in multi-dimensional space |
| **Vector** | Array of numerical values — each number = one feature/attribute dimension |
| **vs Traditional DB** | Traditional = tables + SQL; Vector = embeddings + similarity search |
| **Vector library** | Read/update only; in-memory; no full CRUD |
| **Vector database** | Full CRUD; in-memory or persistent; production-ready |
| **In-memory** | Fastest — best for real-time (RedisAI) |
| **Disk-based** | Large datasets — best for scale (Annoy, Milvus) |
| **Distributed** | Horizontal scaling + fault tolerance (FAISS) |
| **Graph-based** | Relationship and graph analytics (Neo4j) |
| **Time-series** | Temporal pattern analysis (InfluxDB) |
| **Dedicated vector DB** | Built specifically for vectors — uses LSH, product quantization, inverted indexes |
| **DB with vector search** | General DB with vector add-on — more flexible, less optimized |
| **Applications** | Image/video analysis, recommendations, geospatial services, social/marketing |

---

## Similarity Search & Distance Metrics

### What is Similarity Search?

**Similarity search** is the process of finding items in a dataset most similar to a given query. It is the core retrieval operation of every vector database.

**Common applications:**
Recommendation systems · Image and video retrieval · NLP document similarity · Biometrics / face recognition

---

### Vectors — A Quick Refresher

A vector is a geometric object with **length (magnitude)** and **direction**, represented as an array of numbers on a multi-dimensional Cartesian plane.

```
Vector a = [4, 8]
Magnitude ||a|| = √(4² + 8²) ≈ 8.94   (L2 / Euclidean norm)
```

> When used as embeddings:
> - **Direction** → encodes semantic meaning or topic
> - **Magnitude** → can reflect intensity, confidence, or salience (e.g. product popularity)

---

### The Three Core Distance & Similarity Metrics

#### 1. L2 Distance (Euclidean Distance)

Measures the **straight-line distance** between two points in space.

```
L2(a, b) = √ Σ(aᵢ - bᵢ)²
```

**Example** — `a = [4, 8]`, `b = [11.5, 5]`:
```
L2 = √((4-11.5)² + (8-5)²) ≈ 8.08
```

| Property | Detail |
|---|---|
| Sensitive to magnitude | ✅ Yes |
| Sensitive to direction | ✅ Yes |
| Best for | Spatial/geometric data, computer vision, clustering |
| Limitation | Suffers from "curse of dimensionality" in very high dimensions |

---

#### 2. Dot Product (Inner Product) Similarity

Multiplies corresponding elements and sums them — equivalent to multiplying magnitudes accounting for the angle between vectors.

```
a · b = Σ aᵢbᵢ  =  ||a|| × ||b|| × cos(α)
```

**Example** — `a = [4, 8]`, `b = [11.5, 5]`:
```
a · b = (4 × 11.5) + (8 × 5) = 46 + 40 = 86
```

> Dot product is a **similarity** metric — higher value = more similar.
> To convert to a distance metric: use the **negative dot product**.

| Property | Detail |
|---|---|
| Sensitive to magnitude | ✅ Yes |
| Sensitive to direction | ✅ Yes |
| Best for | Neural networks, recommendation systems where popularity matters |
| Note | Higher magnitude vectors (e.g. more popular items) get boosted in ranking |

---

#### 3. Cosine Similarity & Distance

Measures the **angle between two vectors** — ignores magnitude entirely, focuses purely on direction.

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)

cosine_distance(a, b) = 1 - cosine_similarity(a, b)
```

**Shortcut — normalized vectors:**
If vectors are pre-normalized (divided by their L2 norm), cosine similarity reduces to a simple dot product:
```
norm(a) = a / ||a||
cosine_similarity(a, b) = norm(a) · norm(b)
```

> Many embedding models **normalize vectors by default** — making cosine similarity as computationally cheap as a dot product.

| Property | Detail |
|---|---|
| Sensitive to magnitude | ❌ No — direction only |
| Scale-invariant | ✅ Yes |
| Best for | Text embeddings, NLP, high-dimensional sparse data |
| Note | A long and a short document about the same topic score equally |

---

### Metric Comparison

| Metric | Sensitive to Magnitude | Scale-Invariant | Best For |
|---|---|---|---|
| **L2 Distance** | ✅ Yes | ❌ No | Spatial data, clustering, computer vision |
| **Cosine Distance** | ❌ No | ✅ Yes | Text, embeddings, NLP |
| **Dot Product** | ✅ Yes | ❌ No | Neural networks, recommender systems |

---

### Choosing the Right Metric

| Situation | Recommended Metric |
|---|---|
| Spatial or geometric data (images, maps) | **L2 Distance** |
| Text similarity or document comparison | **Cosine Distance** |
| Popularity matters (e.g. trending recommendations) | **Dot Product** |
| Very high-dimensional data | **Cosine Distance** (L2 degrades in high dims) |
| Vectors already normalized | **Dot Product** (computes cosine similarity efficiently) |

---

*Notes based on: Module 01 — Introduction to Vector Databases + Similarity Search Reading (Course 03 — Vector Databases for RAG)*
