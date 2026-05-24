# Course 05 — Module 03: Advanced Multimodal Applications

> A complete reference covering Multimodal RAG (MM-RAG), multimodal chatbots, and QA systems with practical implementation using IBM WatsonX.

---

## Table of Contents

1. [Multimodal RAG (MM-RAG)](#multimodal-rag-mm-rag)
   - [What is MM-RAG?](#what-is-mm-rag)
   - [The MM-RAG Pattern](#the-mm-rag-pattern)
   - [The MM-RAG Pipeline](#the-mm-rag-pipeline)
   - [Case Study: Style Finder](#case-study-style-finder)
2. [Multimodal Chatbots & QA Systems](#multimodal-chatbots--qa-systems)
   - [What are Multimodal Chatbots?](#what-are-multimodal-chatbots)
   - [System Architecture](#system-architecture)
   - [Implementation with Llama 3.2 via IBM WatsonX](#implementation-with-llama-32-via-ibm-watsonx)
3. [Summary](#summary)

---

## Multimodal RAG (MM-RAG)

### What is MM-RAG?

**MM-RAG** (Multimodal Retrieval-Augmented Generation) combines two powerful capabilities:

| Component | Meaning |
|---|---|
| **Multimodal** | Works with multiple data types — images, video, audio, text |
| **Retrieval-Augmented** | Enhances LLM responses by retrieving relevant info from a database |
| **Generation** | Uses retrieved multimodal data to generate detailed, accurate responses |

**Why MM-RAG?** Advanced vision models like Llama 4, GPT-4o, and Claude 3 can see images — but they don't have access to your specific knowledge bases or proprietary databases. MM-RAG bridges this gap by retrieving domain-specific data to ground the model's responses.

---

### The MM-RAG Pattern

MM-RAG follows three core steps:

```
Step 1: Multimodal Data Retrieval
    Retrieve relevant information across modalities
    (text documents, images, audio, video)
        |
Step 2: Contrastive Learning for Embeddings
    Train models to link related data from different types
    Example: image of a cat ↔ "a domestic feline" → mapped to similar representations
        |
Step 3: Generation Informed by Multimodal Context
    Use retrieved multimodal data as context for the generative model
    → richer, more grounded outputs
```

> **Contrastive learning** teaches the system to connect images and text by mapping related concepts from different modalities to similar vector representations in shared embedding space.

---

### The MM-RAG Pipeline

```
1. Data Indexing
   Convert text, images, audio, video → embeddings
   Store in a vector database for efficient retrieval
        |
2. Data Retrieval
   User query (text, image, or both) → converted to embedding
   Semantically search vector database across all modalities
   Return most relevant multimodal data
        |
3. Augmentation
   Combine retrieved multimodal data + original user query
   → enriched context for the generative model
        |
4. Response Generation
   Augmented input → multimodal generative model
   → response integrating information from all retrieved modalities
```

---

### Case Study: Style Finder

**Style Finder** is a practical MM-RAG application that allows users to upload outfit images and receive detailed clothing information with purchase links.

**Pipeline breakdown:**

```
Step 1: Image Encoding
    Upload image → ResNet50 (pretrained, torchvision)
    → feature vector + Base64 string

Step 2: Similarity Search
    Compare image vector against pre-encoded dataset vectors
    using cosine similarity
    → select highest-scoring match
    → retrieve all items from the same outfit

Step 3: Structured Data Retrieval
    Fetch metadata: product names, prices, URLs
    → format for inclusion in LLM prompt

Step 4: Multimodal LLM Call (Llama Vision Instruct)
    Send: structured prompt + Base64-encoded image
    Prompt includes:
      - Professional context (catalog-style analysis)
      - Instructions to describe materials, patterns, colors
      - Item details or similar items (based on similarity score)
    → model returns markdown-compatible structured response

Output: Visual reasoning + retrieved metadata → rich fashion analysis
```

| Component | Technology Used |
|---|---|
| Image encoding | ResNet50 (torchvision) |
| Similarity metric | Cosine similarity |
| LLM | Llama Vision Instruct (via IBM WatsonX) |
| Image format for LLM | Base64 encoded string |

---

## Multimodal Chatbots & QA Systems

### What are Multimodal Chatbots?

**Multimodal chatbots and QA systems** are advanced AI applications that process, understand, and respond to **multiple types of data inputs** — text, images, audio, and sometimes video.

> Unlike traditional text-only chatbots, these systems can **see, read, and understand** the world more like humans do.

**Key features:**

| Feature | Description |
|---|---|
| **Multiple input modalities** | Text, images, audio, video — all accepted as input |
| **Integrated understanding** | Each modality processed separately, then fused for unified context |
| **Contextual response generation** | Outputs range from text answers to action prompts or image generation |

---

### System Architecture

```
Inputs:
  Text (queries, commands, conversations)
  Images (photos, diagrams, screenshots)
  Audio (voice commands, ambient sounds)
  Video (motion-based visual content)
        |
Processing:
  Each modality processed by specialized encoder
        |
Fusion:
  Cross-modal fusion → unified understanding
        |
Response Generation:
  Text answers / suggestions / action prompts / image generation
```

**Example:** A question about an image → text query + visual analysis combined → context-aware response.

---

### Implementation with Llama 3.2 via IBM WatsonX

**Model:** Meta Llama 3.2 90B Vision Instruct via IBM WatsonX

#### Step 1 — Setup & Model Initialization

```python
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

# Authenticate
credentials = Credentials(url="YOUR_URL", api_key="YOUR_API_KEY")
client = APIClient(credentials)

# Initialize model
model = ModelInference(
    model_id="meta-llama/llama-3-2-90b-vision-instruct",
    api_client=client,
    project_id="YOUR_PROJECT_ID",
    params={Params.TEMPERATURE: 0.7}  # customize creativity/determinism
)
```

#### Step 2 — Image Preparation Functions

```python
import base64
import requests

# For local image files
def prepareImage(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# For online images
def prepareImageFromURL(image_url):
    response = requests.get(image_url)
    return base64.b64encode(response.content).decode("utf-8")
```

> Base64 encoding transforms binary image data into a text string that preserves all visual information while making it compatible with text-based APIs.

#### Step 3 — Multimodal Query Function

```python
def queryMultimodalModel(encoded_image, question, system_prompt=None):
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message with text + image
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }}
        ]
    })

    # Send to model and extract response
    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]
```

#### Step 4 — Using the QA System

```python
# Prepare image
encoded_image = prepareImage("workplace.jpg")
# or: encoded_image = prepareImageFromURL("https://example.com/image.jpg")

# Define question
question = "Is there anything unsafe in this workplace photo?"

# Define system prompt (shapes the model's role and perspective)
system_prompt = "You are a workplace safety expert. Identify any hazards visible in the image."

# Query the model
answer = queryMultimodalModel(encoded_image, question, system_prompt)
print(answer)
```

**The power of system prompts:**

| System Prompt | Effect |
|---|---|
| "Act as an expert nutritionist analyzing food" | Focuses on nutritional content, health implications |
| "Act as a fashion consultant giving style advice" | Focuses on clothing, colors, style recommendations |
| "Act as a workplace safety expert" | Focuses on hazards, compliance, safety risks |

> The same image produces very different outputs depending on the system prompt — giving you full control over the model's perspective and focus.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **MM-RAG** | Combines multimodal inputs with RAG — retrieves domain-specific data to ground LLM responses |
| **Why MM-RAG** | Vision models can see but not access private databases — RAG bridges this gap |
| **Contrastive learning** | Maps related data from different modalities to similar vector representations |
| **MM-RAG pipeline** | Data indexing → retrieval → augmentation → response generation |
| **Style Finder** | MM-RAG app: ResNet50 image encoding → cosine similarity search → Llama Vision response |
| **Multimodal chatbots** | Process text, images, audio, video — fuse modalities → context-aware responses |
| **Base64 encoding** | Converts binary image data to text-compatible format for LLM APIs |
| **`prepareImage`** | Encodes local image files to Base64 |
| **`prepareImageFromURL`** | Downloads and encodes online images to Base64 |
| **`queryMultimodalModel`** | Combines encoded image + text question + system prompt into structured API message |
| **System prompt** | Sets the model's role and perspective — dramatically changes output style and focus |
| **Model used** | Meta Llama 3.2 90B Vision Instruct via IBM WatsonX |
| **Temperature parameter** | Controls creativity vs. determinism of model responses |

---

*Notes based on: Course 05 Module 03 — Advanced Multimodal Applications: MM-RAG + Multimodal Chatbots & QA Systems*
