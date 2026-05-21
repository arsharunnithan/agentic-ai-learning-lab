import gradio as gr
from groq import Groq

client = Groq(api_key="YOUR_GROQ_KEY_HERE")

def chat(message, history):
    messages = []
    for item in history:
        messages.append({
            "role": item["role"],
            "content": item["content"]
        })
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    title="My AI Chatbot",
    description="Powered by Llama 3.1 via Groq — Ask me anything!"
)

demo.launch()
