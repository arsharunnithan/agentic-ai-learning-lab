# Module 03 — Gradio Hands-On

Practical Gradio apps built while learning to create AI interfaces from scratch.

## What I Built

| App | File | Description |
|-----|------|-------------|
| App 1 | app1_greeter.py | Basic Gradio interface — core structure |
| App 2 | app2_sentiment.py | Sentiment analysis using HuggingFace |
| App 3 | app3_toolkit.py | Multi-task toolkit with dropdown + GPT-2 |
| App 4 | app4_chatbot.py | AI chatbot with memory via Groq + Llama 3.1 |

## Setup

Install dependencies:
pip install -r requirements.txt

Run any app:
python app1_greeter.py

## Key Concepts Learned
- Gradio Interface vs ChatInterface
- Connecting local HuggingFace models
- Connecting to external AI APIs (Groq)
- Adding conversation memory to a chatbot
- Debugging common setup errors on Windows

## Tools & Libraries
- Gradio — web UI framework
- HuggingFace Transformers — local AI models
- Groq API — fast free LLM inference
- Llama 3.1 — the underlying language model

## Note
Add your Groq API key in app4_chatbot.py before running.
Never commit real API keys to GitHub.
