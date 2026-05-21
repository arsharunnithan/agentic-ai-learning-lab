import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import tempfile
import os

# Step 1 — Configure LLM and Embedding model
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key="YOUR_GROQ_KEY_HERE"
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def generate_icebreakers(pdf_file):
    if pdf_file is None:
        return "Please upload a LinkedIn PDF first!"

    # Step 2 — Save uploaded file temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "profile.pdf")
        with open(pdf_path, "wb") as f:
            with open(pdf_file, "rb") as src:
                f.write(src.read())

        # Step 3 — Load and chunk with LlamaIndex
        documents = SimpleDirectoryReader(tmpdir).load_data()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        # Step 4 — Build Vector Index
        index = VectorStoreIndex(nodes)
        query_engine = index.as_query_engine()

        # Step 5 — Generate icebreakers
        response = query_engine.query("""
        Based on this LinkedIn profile, generate 5 unique and personalized 
        conversation icebreakers. Each icebreaker should:
        - Be specific to this person's career, skills, or achievements
        - Feel natural and genuine, not generic
        - Be a question or conversation starter
        - Reference something specific from their profile
        
        Format as:
        1. [icebreaker]
        2. [icebreaker]
        3. [icebreaker]
        4. [icebreaker]
        5. [icebreaker]
        """)

    return str(response)

with gr.Blocks() as demo:
    gr.Markdown("# 🤝 AI Icebreaker Bot")
    gr.Markdown("Upload a LinkedIn profile PDF and get personalized conversation starters!")

    with gr.Row():
        pdf_input = gr.File(
            label="Upload LinkedIn Profile PDF",
            file_types=[".pdf"]
        )

    generate_btn = gr.Button("✨ Generate Icebreakers", variant="primary")

    output = gr.Textbox(
        label="Your Personalized Icebreakers",
        lines=15,
        placeholder="Icebreakers will appear here..."
    )

    generate_btn.click(
        fn=generate_icebreakers,
        inputs=pdf_input,
        outputs=output
    )

demo.launch()
