import chromadb
from chromadb.utils import embedding_functions
import gradio as gr

# Step 1 — Set up ChromaDB client
client = chromadb.Client()

# Step 2 — Set up embedding function using SentenceTransformer
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Step 3 — Create a collection (like a table in a database)
collection = client.create_collection(
    name="my_documents",
    embedding_function=embedding_fn
)

# Step 4 — Sample documents to store
documents = [
    "Artificial intelligence is transforming the world of technology.",
    "Machine learning allows computers to learn from data.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Data science combines statistics and programming to extract insights.",
    "Python is the most popular language for AI and data science.",
    "Kerala is a beautiful state in southern India known for backwaters.",
    "Biryani is one of the most popular dishes in Indian cuisine.",
]

# Step 5 — Add documents to ChromaDB with unique IDs
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✅ Added {len(documents)} documents to ChromaDB!")

# Step 6 — Similarity search function
def similarity_search(query):
    if not query.strip():
        return "Please enter a search query!"
    
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    output = f"🔍 Top 3 results for: '{query}'\n\n"
    for i, (doc, distance) in enumerate(zip(
        results["documents"][0],
        results["distances"][0]
    )):
        similarity = round((1 - distance) * 100, 1)
        output += f"**Result {i+1}** (Similarity: {similarity}%)\n"
        output += f"{doc}\n\n"
    
    return output

# Step 7 — Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 ChromaDB Similarity Search")
    gr.Markdown("Type anything and find the most similar documents in the database!")

    with gr.Row():
        gr.Markdown("### 📚 Documents in Database:")
    
    for i, doc in enumerate(documents):
        gr.Markdown(f"**{i+1}.** {doc}")

    gr.Markdown("---")
    
    query_input = gr.Textbox(
        label="Enter your search query",
        placeholder="e.g. how do machines learn?"
    )
    search_btn = gr.Button("🔍 Search", variant="primary")
    output = gr.Textbox(label="Results", lines=10)

    search_btn.click(fn=similarity_search, inputs=query_input, outputs=output)

demo.launch()
