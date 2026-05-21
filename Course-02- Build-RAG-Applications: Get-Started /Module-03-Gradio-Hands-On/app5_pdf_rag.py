import gradio as gr
from groq import Groq
from pypdf import PdfReader

client = Groq(api_key="YOUR_GROQ_KEY_HERE")

def load_pdf(pdf_file):
    """Extract all text from a PDF file"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chat(message, history, pdf_file):
    # Step 1 — Load PDF if uploaded
    if pdf_file is not None:
        knowledge_base = load_pdf(pdf_file)
    else:
        knowledge_base = "No document uploaded yet."

    # Step 2 — Inject PDF as context (Naive RAG)
    system_prompt = f"""You are a helpful assistant.
Answer questions ONLY based on the following document:

{knowledge_base}

If the answer is not in the document, say "I don't find that in the document."
"""
    messages = [{"role": "system", "content": system_prompt}]

    for item in history:
        messages.append({
            "role": item["role"],
            "content": item["content"]
        })

    messages.append({"role": "user", "content": message})

    # Step 3 — Generate answer from context
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.File(label="Upload a PDF", file_types=[".pdf"])
    ],
    title="Chat with your PDF",
    description="Upload any PDF and ask questions about it!"
)

demo.launch()
