import chromadb
from chromadb.utils import embedding_functions

# Setup ChromaDB with persistent storage
client = chromadb.PersistentClient(path="./food_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create collection safely
collection = client.get_or_create_collection(
    name="foods",
    embedding_function=ef
)

print("Collection ready:", collection.name)
print("Total foods:", collection.count())

foods = [
    {
        "id": "food_001",
        "name": "Chicken Biryani",
        "cuisine": "Indian",
        "calories": 450,
        "taste": "spicy, savory",
        "ingredients": "chicken, basmati rice, spices, onion, yogurt",
        "cooking_method": "slow cooked"
    },
    {
        "id": "food_002",
        "name": "Margherita Pizza",
        "cuisine": "Italian",
        "calories": 300,
        "taste": "savory, mild",
        "ingredients": "dough, tomato sauce, mozzarella, basil",
        "cooking_method": "baked"
    },
    {
        "id": "food_003",
        "name": "Caesar Salad",
        "cuisine": "American",
        "calories": 180,
        "taste": "savory, fresh",
        "ingredients": "romaine lettuce, croutons, parmesan, caesar dressing",
        "cooking_method": "raw"
    },
    {
        "id": "food_004",
        "name": "Tacos al Pastor",
        "cuisine": "Mexican",
        "calories": 350,
        "taste": "spicy, savory",
        "ingredients": "pork, pineapple, corn tortilla, onion, cilantro",
        "cooking_method": "grilled"
    },
    {
        "id": "food_005",
        "name": "Chocolate Lava Cake",
        "cuisine": "French",
        "calories": 420,
        "taste": "sweet, rich",
        "ingredients": "chocolate, butter, eggs, sugar, flour",
        "cooking_method": "baked"
    },
    {
        "id": "food_006",
        "name": "Grilled Salmon",
        "cuisine": "American",
        "calories": 280,
        "taste": "savory, fresh",
        "ingredients": "salmon, lemon, garlic, olive oil, herbs",
        "cooking_method": "grilled"
    },
    {
        "id": "food_007",
        "name": "Pad Thai",
        "cuisine": "Thai",
        "calories": 380,
        "taste": "spicy, sweet, savory",
        "ingredients": "rice noodles, shrimp, peanuts, egg, bean sprouts",
        "cooking_method": "stir fried"
    },
    {
        "id": "food_008",
        "name": "Tiramisu",
        "cuisine": "Italian",
        "calories": 350,
        "taste": "sweet, creamy",
        "ingredients": "mascarpone, espresso, ladyfingers, cocoa, eggs",
        "cooking_method": "chilled"
    }
]

print("Number of foods:", len(foods))

def food_to_text(food):
    return f"{food['name']} is a {food['cuisine']} dish that tastes {food['taste']}, \
made with {food['ingredients']}, cooked by {food['cooking_method']} \
with {food['calories']} calories."

print(food_to_text(foods[0]))

# Convert all foods to text
documents = [food_to_text(food) for food in foods]
ids = [food['id'] for food in foods]
metadatas = foods

# Add to ChromaDB only if empty
if collection.count() == 0:
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Added {len(foods)} foods to ChromaDB!")
else:
    print(f"Collection already has {collection.count()} foods!")


import gradio as gr
from groq import Groq

groq_client = Groq(api_key="YOUR GROQ KEY")

# ─── Part 1: Basic Similarity Search ───
def basic_search(query):
    results = collection.query(query_texts=[query], n_results=3)
    output = f"🍽️ Top matches for: '{query}'\n\n"
    for i, (metadata, distance) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = round((1 - distance) * 100, 1)
        output += f"Rank {i+1} — {metadata['name']} ({similarity}% match)\n"
        output += f"Cuisine: {metadata['cuisine']} | Calories: {metadata['calories']}\n"
        output += f"Taste: {metadata['taste']}\n"
        output += f"Ingredients: {metadata['ingredients']}\n\n"
    return output

# ─── Part 2: Advanced Filtering ───
def filtered_search(query, cuisine_filter, max_calories):
    where_filter = {}
    
    if cuisine_filter and cuisine_filter != "Any":
        where_filter["cuisine"] = {"$eq": cuisine_filter}
    
    if max_calories:
        where_filter["calories"] = {"$lte": int(max_calories)}
    
    try:
        if where_filter:
            results = collection.query(
                query_texts=[query],
                n_results=3,
                where=where_filter if len(where_filter) == 1 else {"$and": [
                    {k: v} for k, v in where_filter.items()
                ]}
            )
        else:
            results = collection.query(query_texts=[query], n_results=3)
        
        if not results["documents"][0]:
            return "No results found with these filters!"
        
        output = f"🔍 Filtered results for: '{query}'\n\n"
        for i, (metadata, distance) in enumerate(zip(
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = round((1 - distance) * 100, 1)
            output += f"Rank {i+1} — {metadata['name']} ({similarity}% match)\n"
            output += f"Cuisine: {metadata['cuisine']} | Calories: {metadata['calories']}\n"
            output += f"Taste: {metadata['taste']}\n\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# ─── Part 3: RAG Chatbot ───
def rag_chat(message, history):
    # Retrieve relevant foods
    results = collection.query(query_texts=[message], n_results=3)
    
    context = ""
    for metadata in results["metadatas"][0]:
        context += f"- {metadata['name']} ({metadata['cuisine']}, {metadata['calories']} cal, {metadata['taste']})\n"
    
    # Build messages
    messages = [{"role": "system", "content": f"""You are a friendly food recommendation assistant.
Use these relevant foods from our database to answer:

{context}

Give personalized recommendations based on these options."""}]
    
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    
    messages.append({"role": "user", "content": message})
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content

# ─── Gradio UI ───
with gr.Blocks() as demo:
    gr.Markdown("# 🍕 AI Food Recommendation System")
    
    with gr.Tab("🔍 Basic Search"):
        gr.Markdown("### Find foods by description")
        query1 = gr.Textbox(label="What are you craving?", 
                           placeholder="e.g. something spicy and filling")
        btn1 = gr.Button("Search", variant="primary")
        out1 = gr.Textbox(label="Results", lines=12)
        btn1.click(fn=basic_search, inputs=query1, outputs=out1)
    
    with gr.Tab("🎯 Filtered Search"):
        gr.Markdown("### Search with filters")
        query2 = gr.Textbox(label="What are you craving?",
                           placeholder="e.g. healthy and fresh")
        cuisine = gr.Dropdown(
            choices=["Any", "Indian", "Italian", "American", "Mexican", "French", "Thai"],
            label="Cuisine", value="Any"
        )
        calories = gr.Textbox(label="Max Calories (leave blank for no limit)",
                             placeholder="e.g. 300")
        btn2 = gr.Button("Search", variant="primary")
        out2 = gr.Textbox(label="Results", lines=12)
        btn2.click(fn=filtered_search, inputs=[query2, cuisine, calories], outputs=out2)
    
    with gr.Tab("🤖 Food Chatbot"):
        gr.Markdown("### Chat with your AI food assistant")
        gr.ChatInterface(fn=rag_chat)

demo.launch()
