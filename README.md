# Agentic AI Learning Lab

A structured learning journal documenting my progress through the **IBM RAG and Agentic AI Professional Certificate** — from Generative AI foundations to building production-ready agentic applications with LangChain, LangGraph, and Flask.

---

## 🎯 Goals

- Understand Generative AI and Large Language Model fundamentals
- Master Prompt Engineering techniques
- Build modular AI systems using LangChain
- Develop and deploy production-ready AI applications with Flask
- Build RAG pipelines with vector databases
- Create multimodal AI applications
- Design and deploy multi-agent systems with LangGraph

---

## 📂 Repository Structure

```
agentic-ai-learning-lab/
├── Course-01-Develop-GenAI-Applications/
│   ├── Module-01-GenAI-Prompt-Engineering/
│   ├── Module-02-Introduction-to-LangChain/
│   └── Module-03-Building-with-Flask/
├── Course-02-Build-RAG-Applications-Get-Started/
│   ├── Module-01-Intro to RAG.md
│   ├── Module-02-Intro-to-Gradio.md
│   ├── Module-03-Gradio-Hands-On/
│   │   ├── Readme.md
│   │   ├── app1_greeter.py
│   │   ├── app2_sentiment.py
│   │   ├── app3_toolkit.py
│   │   ├── app4_chatbot.py
│   │   ├── app5_pdf_rag.py
│   │   └── requirements.txt
│   ├── Module-04-LangChain-QA-Bot/
│   │   ├── README.md
│   │   ├── qa_bot.py
│   │   └── requirements.txt
│   ├── Module-05-Build-RAG-Apps-with-LlamaIndex.md
│   └── Module-06-IcebreakerBot/
│       ├── ReadMe.md
│       ├── icebreaker_bot.py
│       └── requirements.txt
├── Course-03-Vector-Databases-for-RAG-An-Introduction/
│   ├── Module-01-Introduction to Vector Databases and Similarity Search.md
│   ├── Module-02-Exploring chroma DB.md
│   ├── Module-03-HandsOn-ChromaDB/
│   │   ├── Readme.md
│   │   ├── chroma_similarity.py
│   │   ├── employee_search.py
│   │   └── requirements.txt
│   └── Module-04-Vector Databases for Recommendation systems and RAG.md
├── Couse-04-Advanced-RAG-with-Vector-Databases-and-Retrievers/
│   ├── Module-01-Advanced Retrivers for RAG.md
│   └── Module-02-FAISS, HNSW & Comprehensive RAG Applications.md
├── Course-05-Build-Multimodal-Generative-AI-Applications/
│   ├── Module-01-Introduction to Multimodal AI: Text and Speech Processing.md
│   ├── Module-02-Integrating Visual and Video Modalities.md
│   └── Module-03-Advanced Multimodal Applications.md
├── Course-06-Fundamentals-of-Building-AI-Agents/
│   ├── Module-01-Foundations of Tool Calling and Chaining.md
│   ├── Module-02-LCEL and Manual Tool Calling in LangChain.md
│   └── Module-03-Using Built-in Agents in LangChain.md
├── Course-07-Agentic-AI-with-LangChain-and-LangGraph/
│   ├── Module-01-Introduction to LangGraph.md
│   ├── Module-02-Build Self-Improving Agents with LangGraph.md
│   └── Module-03-Multi-Agent Systems and Agentic RAG with LangGraph.md
└── README.md
```

---

## 📚 Courses & Modules

### ✅ Course 01 — Develop Generative AI Applications

#### Module 01 — Foundations of Generative AI & Prompt Engineering

| Topic | Description |
|---|---|
| Generative AI Models | Foundation models, next-token prediction, use cases |
| Natural Language Processing | NLU, NLG, tokenization, NER, stemming |
| Core GenAI Concepts | RAG, agents, vector databases, fine-tuning, hallucination mitigation |
| In-Context Learning | Task adaptation via prompt examples without retraining |
| Prompt Engineering | Zero-shot, one-shot, few-shot, Chain-of-Thought, self-consistency |
| LangChain LCEL | Pipe-based chaining, RunnableParallel, RunnableLambda, type coercion |

#### Module 02 — Introduction to LangChain in GenAI Applications

| Topic | Description |
|---|---|
| LangChain Overview | Framework purpose, modularity, extensibility, decomposition |
| Core Components | Language models, chat models, prompt templates, output parsers |
| Chains & Memory | Sequential chains, ChatMessageHistory, context preservation |
| Agents | LLM-driven autonomous systems, tool use, Pandas DataFrame agent |
| LCEL Deep Dive | Runnable primitives, parallel execution, async support, tracing |

#### Module 03 — Building a Generative AI Application with Flask

| Topic | Description |
|---|---|
| Choosing the Right AI Model | Multi-model approach, evaluation criteria, MLOps governance |
| Introduction to Flask | Micro framework, routing, Jinja2, extensions, vs Django |
| Flask for Large Scale Projects | Scalability patterns, modular design, real-world usage |
| From Idea to AI | Developer journey — ideation, RAG vs fine-tuning, MLOps |

---

### ✅ Course 02 — Build RAG Applications: Get Started

#### Module 01 — Introduction to RAG

| Topic | Description |
|---|---|
| What is RAG | Combining retrieval with generation for grounded responses |
| RAG vs Long Context | Why RAG beats pure long-context approaches |
| RAG Pipeline | 8-step process from document indexing to response generation |
| Embeddings & Retrieval | Tokenization, chunking, vector averaging, distance metrics |

#### Module 02 — Introduction to Gradio

| Topic | Description |
|---|---|
| Gradio Basics | `gr.Interface`, inputs, outputs, launch |
| Components | Textbox, Number, Slider, Image, File, Label |
| Image Captioning | BLIP model integration |
| Image Classification | ResNet-18 with PyTorch and softmax |

#### Module 03 — Gradio Hands-On Projects

| App | Description |
|---|---|
| `app1_greeter.py` | Basic text input/output interface |
| `app2_sentiment.py` | Sentiment analysis with LLM |
| `app3_toolkit.py` | Multi-tool interface |
| `app4_chatbot.py` | Conversational chatbot with history |
| `app5_pdf_rag.py` | Full RAG pipeline on uploaded PDFs |

#### Module 04 — LangChain QA Bot

| Topic | Description |
|---|---|
| Document loading & chunking | LangChain loaders and text splitters |
| Embedding & retrieval | Vector store integration |
| QA chain | RetrievalQA with custom prompts |

#### Module 05 — Build RAG Apps with LlamaIndex

| Topic | Description |
|---|---|
| LlamaIndex pipeline | SimpleDirectoryReader, VectorStoreIndex, query engine |
| IBM Granite integration | Using Granite LLM with LlamaIndex |
| LlamaIndex vs LangChain | Comparison across loading, chunking, embedding, retrieval |

#### Module 06 — Icebreaker Bot

| Topic | Description |
|---|---|
| Icebreaker bot | Conversational bot grounded in document context using LlamaIndex |
| IBM Granite integration | Granite LLM + LlamaIndex + vector store |

---

### ✅ Course 03 — Vector Databases for RAG: An Introduction

#### Module 01 — Introduction to Vector Databases and Similarity Search

| Topic | Description |
|---|---|
| What is a Vector Database | High-dimensional vector storage, similarity search, nearest neighbor queries |
| Vectors | Arrays of numerical values representing data features and attributes |
| Vector DB vs Traditional DB | Tables/SQL vs embeddings/similarity search — indexing, scalability, applications |
| Vector Libraries vs Vector DBs | CRUD capabilities, in-memory vs persistent, production readiness |
| Types of Vector DBs | In-memory, disk-based, distributed, graph-based, time-series |
| Dedicated vs Vector Search DBs | Purpose-built vs general DBs with add-ons (LSH, product quantization) |
| Similarity Search & Distance Metrics | L2 distance, dot product, cosine similarity — when to use each |
| Real-World Applications | Image/video analysis, recommendations, geospatial, social/marketing |

#### Module 02 — Exploring Chroma DB

| Topic | Description |
|---|---|
| ChromaDB Capabilities | Vector search, full-text search, metadata filtering, multi-modal retrieval |
| Deployment Options | Client-server vs standalone mode |
| Architecture & Workflow | 5-phase pipeline from embedding to querying |
| Metadata Filtering | `where` parameter — `$eq`, `$ne`, `$gt`, `$lt`, `$in`, `$and`, `$or` |
| Document Filtering | `where_document` — `$contains`, `$not_contains` |
| HNSW Algorithm | Multi-layered graph ANN — `ef_search`, `ef_construction`, `max_neighbors` |
| ChromaDB Operations | `create_collection`, `add`, `get`, `update`, `delete`, `query` |

#### Module 03 — Hands-On ChromaDB

| File | Description |
|---|---|
| `chroma_similarity.py` | Similarity search implementation with ChromaDB |
| `employee_search.py` | Semantic employee search application |

#### Module 04 — Vector Databases for Recommendation Systems and RAG

| Topic | Description |
|---|---|
| Vector DBs in RAG | How vector databases power the retrieval step in RAG pipelines |
| Recommendation Systems | Embedding-based similarity for personalized recommendations |
| Real-World Applications | E-commerce, streaming, enterprise knowledge management |

---

### ✅ Couse 04 — Advanced RAG with Vector Databases and Retrievers

#### Module 01 — Advanced Retrievers for RAG

| Topic | Description |
|---|---|
| LangChain Retrievers | Vector store-based, similarity search, MMR |
| Multi-Query Retriever | LLM generates query variants → unique union of results |
| Self-Query Retriever | Splits query into semantic search + metadata filter |
| Parent Document Retriever | Small chunks for embedding, large parent chunks returned |
| LlamaIndex Index Types | VectorStoreIndex, DocumentSummaryIndex, KeywordTableIndex |
| BM25 Retriever | Keyword-based — TF-IDF improvements, term frequency saturation |
| Auto Merging Retriever | Hierarchical chunks — returns parent if enough children match |
| Query Fusion Retriever | Combines retrievers — Reciprocal Rank, Relative Score, Distribution-Based Fusion |

#### Module 02 — FAISS, HNSW & Comprehensive RAG Applications

| Topic | Description |
|---|---|
| FAISS Overview | Library for fast local vector search — full control, no metadata |
| FAISS vs ChromaDB vs Milvus | Library vs full DB vs distributed production DB |
| FAISS Index Types | Flat (exact), IVF (clustered), LSH (hash-based), HNSW (graph-based) |
| Milvus | FAISS + metadata + distributed scaling |
| HNSW Deep Dive | Multi-layer graph, greedy routing, layer structure, build parameters |
| HNSW Parameters | M (connections), efConstruction (build quality), efSearch (query accuracy) |

---

### ✅ Course 05 — Build Multimodal Generative AI Applications

#### Module 01 — Introduction to Multimodal AI: Text and Speech Processing

| Topic | Description |
|---|---|
| What is Multimodal AI | Processes text, images, audio, video simultaneously |
| Evolution of Multimodal AI | From siloed CNNs/Transformers → CLIP → unified models |
| 5-Component Architecture | Input processing → feature extraction → alignment → fusion → output |
| Computer Vision | CNNs, image captioning, VQA, document analysis |
| STT Pipeline | Audio preprocessing, acoustic model, language model, Wave2Vec2 |
| TTS Pipeline | Text preprocessing, acoustic model, neural vocoder, VITS end-to-end |
| Fusion Strategies | Late fusion vs early fusion — cross-attention mechanisms |
| Challenges | Hallucinations, bias, deepfakes, privacy, compute cost, explainability |

#### Module 02 — Integrating Visual and Video Modalities

| Topic | Description |
|---|---|
| Image Captioning | 3-stage pipeline — input processing, encoding, multimodal LLM processing |
| Llama 4 via IBM WatsonX | Visual encoder + fusion layer + language generation |
| Text-to-Video | Diffusion models, 3D U-Nets, temporal consistency, frame interpolation |
| Image-to-Video | Optical flow, GANs/VAEs, video assembly |
| OpenAI Sora | Diffusion-based Transformer — Remix, Blend, Loop editing tools |
| Vision Model Strengths | Zero-shot learning, cross-modal understanding, robustness |
| Vision Model Limitations | Hallucinations, compute cost, alignment issues |
| Case Studies | Envision (accessibility), Waymo EMMA (autonomous driving), PathChat (healthcare) |

#### Module 03 — Advanced Multimodal Applications

| Topic | Description |
|---|---|
| MM-RAG | Multimodal RAG — retrieval across text, images, audio, video |
| Contrastive Learning | Maps related data from different modalities to similar vector representations |
| MM-RAG Pipeline | Data indexing → retrieval → augmentation → response generation |
| Style Finder | ResNet50 encoding → cosine similarity → Llama Vision response |
| Multimodal Chatbots | Text + image + audio input → fused understanding → context-aware response |
| Base64 Encoding | Converts binary images to text-compatible format for LLM APIs |
| System Prompts | Sets model role and perspective — dramatically changes output style |

---

### ✅ Course 06 — Fundamentals of Building AI Agents

#### Module 01 — Foundations of Tool Calling and Chaining

| Topic | Description |
|---|---|
| Compound AI Systems | Modular systems combining models, tools, databases, and APIs |
| AI Agents | LLM controls the logic — reason, act, memory |
| ReAct Framework | Thought → Action → Observation loop |
| AI System Paradigms | Single LLM vs Structured Workflow vs Autonomous Agent |
| 4-Criteria Framework | Task ambiguity, cost value, capability test, failure impact |
| Tool Calling | LLM generates structured JSON → external system executes |
| Traditional vs Embedded | Embedded tool calling prevents hallucination via library retries |
| @tool Decorator | Preferred syntax — typed, multi-input, structured tools |
| LangChain Agent Types | zero-shot-react, structured-chat-zero-shot-react, openai-functions |
| LangGraph vs initialize_agent | LangGraph preferred for robust multi-step workflows |
| Built-in Tools | 50+ tools: SerpAPI, Python REPL, SQL DB, Gmail, GitHub, DALL-E |

#### Module 02 — LCEL and Manual Tool Calling

| Topic | Description |
|---|---|
| LCEL | Pipe-based composable chains — `\|` operator |
| Runnable Primitives | RunnableSequence, RunnableParallel, RunnableLambda |
| Type Coercion | Dicts → RunnableParallel; functions → RunnableLambda (auto) |
| Manual Tool Calling | Safety, cost control, accuracy — human reviews before execution |
| Structured Outputs | Pydantic models + `with_structured_output()` |
| HumanMessage / AIMessage / ToolMessage | Chat history message types |
| tool_call_id | Links tool results back to requests — critical for multiple calls |
| ToolCallingAgent Class | Encapsulates full manual tool-calling workflow |

#### Module 03 — Using Built-in Agents in LangChain

| Topic | Description |
|---|---|
| Pandas DataFrame Agent | Natural language → Python code against a DataFrame |
| `create_pandas_dataframe_agent` | Passes LLM + DataFrame; auto-generates and executes code |
| Data Visualization | Ask in plain English — agent generates matplotlib/seaborn code |
| AI-Powered SQL Agent | Natural language → SQL → database → natural language response |
| `create_sql_agent` | Connects LLM to MySQL/SQLite via SQLDatabase |
| Natural Language Interfaces | One-shot vs conversational NLI — rule-based vs ML vs hybrid approaches |
| NLI Challenges | Ambiguity, schema mapping, query complexity, domain variation |

---

### ✅ Course 07 — Agentic AI with LangChain and LangGraph

#### Module 01 — Introduction to LangGraph

| Topic | Description |
|---|---|
| Generative vs Agentic AI | Reactive (generates) vs proactive (perceive → decide → act → learn) |
| Chain-of-Thought Reasoning | LLM breaks complex tasks into smaller logical steps |
| AI Agents vs Agentic AI | Single entity vs multi-agent collaborative team |
| Memory Types | Episodic, semantic, vector memory |
| LangGraph Core Primitives | Nodes (functions), Edges (flow), State (shared memory) |
| LangGraph Capabilities | Looping, branching, state persistence, human-in-the-loop, time travel |
| LangGraph vs LangChain | Graph (loops) vs DAG (linear); robust state vs pass-through |
| LangGraph Workflow | StateGraph → add_node → add_edge → set_entry_point → compile → invoke |
| Conditional Edges | `add_conditional_edges` — dynamic routing based on state |

#### Module 02 — Build Self-Improving Agents

| Topic | Description |
|---|---|
| 5 Types of AI Agents | Simple Reflex, Model-Based, Goal-Based, Utility-Based, Learning |
| Reflection Agents | Generator + Reflector loop — iteratively refines output |
| MessageGraph | Specialized StateGraph — state is a list of accumulated messages |
| Reflexion Agents | Reflection + external tools + citations + verifiable claims |
| AnswerQuestion Schema | Pydantic schema: answer, reflection, search_queries |
| ReviseAnswer Schema | Extends AnswerQuestion with references and citations |
| Pydantic for Tool Calls | Structured, validated, JSON-serializable LLM outputs |
| `Literal` Type | Restricts field to specific constant values |
| ReAct Agents | Thought → Action → Action Input → Observation → Final Answer |
| `ToolNode` | Pre-built LangGraph node that executes tool calls from AIMessage |

#### Module 03 — Multi-Agent Systems and Agentic RAG

| Topic | Description |
|---|---|
| Multi-Agent Systems | Multiple autonomous agents collaborating toward collective goals |
| Agent Specialization | Capability boundaries, interface standardization, handoff patterns |
| Collaboration Patterns | Pipeline, Hub-and-Spoke, Parallel with Aggregation, Interactive Dialogue |
| MCP | Anthropic's Model Context Protocol — JSON-RPC universal connector |
| ACP | IBM's Agent Communication Protocol — secure inter-agent communication |
| Orchestration Frameworks | LangGraph, CrewAI, AutoGen, IBM BeeAI |
| Agentic RAG | Agent decides which database to query — smarter than standard RAG |
| Agentic AI Governance | Model/orchestration/tool layer safeguards, human-in-the-loop, auditability |
| LangGraph Multi-Agent | Shared TypedDict state, agent nodes, routing function, conditional edges |

---

## 📈 Progress

**Course 01 — Develop Generative AI Applications**
- [x] Module 01 — Foundations of Generative AI & Prompt Engineering
- [x] Module 02 — Introduction to LangChain in GenAI Applications
- [x] Module 03 — Building a Generative AI Application with Flask

**Course 02 — Build RAG Applications: Get Started**
- [x] Module 01 — Introduction to RAG
- [x] Module 02 — Introduction to Gradio
- [x] Module 03 — Gradio Hands-On Projects
- [x] Module 04 — LangChain QA Bot
- [x] Module 05 — Build RAG Apps with LlamaIndex
- [x] Module 06 — Icebreaker Bot

**Course 03 — Vector Databases for RAG: An Introduction**
- [x] Module 01 — Introduction to Vector Databases and Similarity Search
- [x] Module 02 — Exploring Chroma DB
- [x] Module 03 — Hands-On ChromaDB
- [x] Module 04 — Vector Databases for Recommendation Systems and RAG

**Couse 04 — Advanced RAG with Vector Databases and Retrievers**
- [x] Module 01 — Advanced Retrievers for RAG
- [x] Module 02 — FAISS, HNSW & Comprehensive RAG Applications

**Course 05 — Build Multimodal Generative AI Applications**
- [x] Module 01 — Introduction to Multimodal AI: Text and Speech Processing
- [x] Module 02 — Integrating Visual and Video Modalities
- [x] Module 03 — Advanced Multimodal Applications

**Course 06 — Fundamentals of Building AI Agents**
- [x] Module 01 — Foundations of Tool Calling and Chaining
- [x] Module 02 — LCEL and Manual Tool Calling
- [x] Module 03 — Using Built-in Agents in LangChain

**Course 07 — Agentic AI with LangChain and LangGraph**
- [x] Module 01 — Introduction to LangGraph
- [x] Module 02 — Build Self-Improving Agents
- [x] Module 03 — Multi-Agent Systems and Agentic RAG

---

## 🛠 Upcoming Implementations

- [ ] Prompt experimentation notebook
- [ ] LangChain sequential + parallel chain demo
- [ ] RAG pipeline with vector database
- [ ] Flask-based AI web application
- [ ] Model comparison mini-dashboard
- [ ] Multi-agent research assistant
- [ ] Agentic RAG with routing logic

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=flat&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat&logo=chainlink&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![IBM](https://img.shields.io/badge/IBM_AI-052FAD?style=flat&logo=ibm&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-7C3AED?style=flat&logoColor=white)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
