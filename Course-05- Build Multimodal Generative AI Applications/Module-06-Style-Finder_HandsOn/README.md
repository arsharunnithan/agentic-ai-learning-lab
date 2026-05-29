# Course 05 — Module 06: Style Finder — Multimodal RAG Fashion Analyst

A production-grade multimodal RAG pipeline that combines computer vision,
vector similarity search, and vision LLM to analyze fashion images and
find visually similar products from a real dataset of 44,000+ items.

## What it does
- Upload any fashion photo (from Pinterest, Instagram, anywhere)
- ResNet50 encodes the image into a 2048-dimensional feature vector
- Cosine similarity search finds top 3 visually similar items
  from a real fashion dataset of 44,072 products
- Llama Vision analyzes the actual image
- Returns professional fashion analysis with styling tips
  and shopping recommendations

## How to run
```bash
pip install -r requirements.txt
python style_finder.py
```

## Tech Stack
- Gradio — professional web interface
- ResNet50 (PyTorch) — image feature extraction
- HuggingFace Datasets — 44,072 real fashion products
- Scikit-learn — cosine similarity search
- Groq API — fast LLM inference
- Llama 4 Scout Vision — multimodal image analysis
- Pickle — embeddings persistence

## The MM-RAG Pipeline
```
User uploads fashion image
        ↓
ResNet50 → 2048-dim feature vector
        ↓
Cosine similarity vs 500 pre-encoded product embeddings
        ↓
Top 3 visually similar items retrieved with metadata
        ↓
Image + retrieved context → Llama Vision
        ↓
Professional fashion analysis + styling tips + shopping recs
```

## Key Concepts Demonstrated
- Image embeddings — ResNet50 converts images to 2048 numbers
- Vector similarity search — cosine similarity for visual matching
- Embedding persistence — encode once, reuse forever (pickle)
- Multimodal RAG — retrieval augmented generation with images
- Vision LLM — AI that actually sees and analyzes images
- Real dataset — 44,072 fashion products from HuggingFace

## Dataset
Uses `ashraq/fashion-product-images-small` from HuggingFace
- 44,072 real fashion product images
- Categories: Apparel, Footwear, Accessories
- Metadata: color, usage, gender, season, article type

## How embeddings work
```
Fashion photo → ResNet50 (pretrained on 1M images)
→ [0.23, 0.87, 0.11, ...] (2048 numbers)
→ Similar visuals = similar numbers
→ Cosine similarity finds closest matches
```

## Performance
- First run: ~3 minutes (encodes 500 items, saves to disk)
- Subsequent runs: instant (loads from fashion_embeddings.pkl)
- Query time: ~2-3 seconds per image

## Note
Add your Groq API key in style_finder.py before running.
Never commit real API keys to GitHub.
