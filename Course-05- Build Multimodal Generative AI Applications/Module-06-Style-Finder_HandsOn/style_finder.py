import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import pickle
from datasets import load_dataset
import io
from groq import Groq
import base64
import io

# Setup Groq
client = Groq(api_key="YOUR_GROQ_KEY_HERE")

print("🔄 Loading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
print("✅ ResNet50 ready!")

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def encode_image(image):
    """Convert PIL image to 2048-dim feature vector"""
    img = image.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(tensor)
    return features.squeeze().numpy()

print("🔄 Loading fashion dataset (this may take a minute)...")
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train"
)
print(f"✅ Dataset loaded! Total items: {len(dataset)}")
print(f"Features: {dataset.features}")
print(f"\nSample item:")
print(f"  Name: {dataset[0]['productDisplayName']}")
print(f"  Category: {dataset[0]['masterCategory']}")
print(f"  Color: {dataset[0]['baseColour']}")
print(f"  Usage: {dataset[0]['usage']}")


# Phase 1 — Encode dataset images
EMBEDDINGS_FILE = "fashion_embeddings.pkl"
# Replace DATASET_SIZE and encoding loop with this:
DATASET_SIZE = 1000

if not os.path.exists(EMBEDDINGS_FILE):
    print(f"🔄 Encoding diverse fashion items...")
    
    # Get indices from different categories for diversity
    categories = {}
    for i in range(len(dataset)):
        cat = dataset[i]["masterCategory"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)
    
    print(f"Categories found: {list(categories.keys())}")
    
    # Sample evenly from each category
    selected_indices = []
    per_category = DATASET_SIZE // len(categories)
    for cat, indices in categories.items():
        selected_indices.extend(indices[:per_category])
    
    embeddings = []
    metadata = []
    
    for i, idx in enumerate(selected_indices):
        try:
            item = dataset[idx]
            vector = encode_image(item["image"])
            embeddings.append(vector)
            metadata.append({
                "id": item["id"],
                "name": item["productDisplayName"],
                "category": item["masterCategory"],
                "subcategory": item["subCategory"],
                "article": item["articleType"],
                "color": item["baseColour"],
                "usage": item["usage"],
                "gender": item["gender"],
                "season": item["season"],
                "dataset_idx": idx
            })
            if (i+1) % 100 == 0:
                print(f"  Encoded {i+1}/{len(selected_indices)}...")
        except Exception as e:
            continue
    
    embeddings = np.array(embeddings)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "metadata": metadata}, f)
    print(f"✅ Encoded {len(embeddings)} diverse items!")
print(f"\nEmbeddings shape: {embeddings.shape}")




from sklearn.metrics.pairwise import cosine_similarity

def find_similar_items(query_image, top_k=3):
    """Find most visually similar items from dataset"""
    
    # Encode the uploaded image
    query_vector = encode_image(query_image).reshape(1, -1)
    
    # Calculate cosine similarity against all 500 embeddings
    similarities = cosine_similarity(query_vector, embeddings)[0]
    
    # Get top_k most similar indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        item = metadata[idx].copy()
        item["similarity_score"] = round(float(similarities[idx]) * 100, 1)
        item["image"] = dataset[idx]["image"]
        results.append(item)
    
    return results

# Test it with first dataset image
print("\n🔍 Testing similarity search...")
test_image = dataset[0]["image"]
similar_items = find_similar_items(test_image, top_k=3)

print(f"\nQuery: {metadata[0]['name']}")
print(f"\nTop 3 similar items:")
for i, item in enumerate(similar_items):
    print(f"\nRank {i+1} — {item['name']} ({item['similarity_score']}% similar)")
    print(f"  Category: {item['category']} | Color: {item['color']}")
    print(f"  Usage: {item['usage']}")



def analyze_fashion(query_image, similar_items):
    """Use Llama 3.2 Vision to analyze fashion"""
    
    # Convert image to base64
    buffer = io.BytesIO()
    query_image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Build context from similar items
    context = "Similar items found in our catalog:\n\n"
    for i, item in enumerate(similar_items):
        context += f"Item {i+1} ({item['similarity_score']}% visual match):\n"
        context += f"  Name: {item['name']}\n"
        context += f"  Category: {item['category']} — {item['article']}\n"
        context += f"  Color: {item['color']}\n"
        context += f"  Usage: {item['usage']}\n"
        context += f"  Gender: {item['gender']}\n"
        context += f"  Season: {item['season']}\n\n"
    
    # Send to Llama 3.2 Vision
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an expert fashion analyst and personal stylist.
Analyze this fashion image and provide a detailed, professional analysis.

{context}

Please provide:

## 👗 Outfit Analysis
Describe what you see in the image — clothing items, colors, style.

## ✨ Style Profile
What style aesthetic does this represent? (e.g. Minimalist, Streetwear, Business Casual)

## 🎯 Similar Items Found
Reference the catalog matches and explain why they're similar.

## 💡 Styling Tips
3 specific tips to style this outfit or similar pieces.

## 🛍️ Shopping Recommendations
Based on the style, suggest where to shop (budget and premium options).

Keep the tone professional but friendly — like a personal stylist."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

print("✅ Vision analysis ready!")





import gradio as gr
import base64
import io

def style_finder(image):
    if image is None:
        return "Please upload an image!", None, None, None
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Step 1 — Find similar items
    print("🔍 Finding similar items...")
    similar_items = find_similar_items(image, top_k=3)
    
    # Step 2 — Analyze with Vision LLM
    print("🤖 Analyzing with Llama Vision...")
    analysis = analyze_fashion(image, similar_items)
    
    # Step 3 — Prepare similar item images for display
    similar_image_1 = similar_items[0]["image"]
    similar_image_2 = similar_items[1]["image"]
    similar_image_3 = similar_items[2]["image"]
    
    # Step 4 — Build match info text
    match_info = "## 🔍 Similar Items Found\n\n"
    for i, item in enumerate(similar_items):
        match_info += f"**Match {i+1} — {item['similarity_score']}% similar**\n"
        match_info += f"📌 {item['name']}\n"
        match_info += f"🏷️ {item['category']} | {item['article']}\n"
        match_info += f"🎨 Color: {item['color']}\n"
        match_info += f"👤 {item['gender']} | {item['usage']}\n\n"
    
    return analysis, match_info, similar_image_1, similar_image_2, similar_image_3

# Professional UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 👗 Style Finder — AI Fashion Analyst
    *Upload any fashion image and get professional style analysis powered by computer vision + AI*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Fashion Image",
                type="pil"
            )
            analyze_btn = gr.Button(
                "✨ Analyze Style",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("### 🔍 Similar Items from Catalog")
            with gr.Row():
                sim_img_1 = gr.Image(label="Match 1", height=150)
                sim_img_2 = gr.Image(label="Match 2", height=150)
                sim_img_3 = gr.Image(label="Match 3", height=150)
            
            match_info_output = gr.Markdown()
        
        with gr.Column(scale=1):
            analysis_output = gr.Markdown(
                label="Fashion Analysis",
                value="*Upload an image and click Analyze Style to get started!*"
            )
    
    analyze_btn.click(
        fn=style_finder,
        inputs=image_input,
        outputs=[
            analysis_output,
            match_info_output,
            sim_img_1,
            sim_img_2,
            sim_img_3
        ]
    )

demo.launch()
