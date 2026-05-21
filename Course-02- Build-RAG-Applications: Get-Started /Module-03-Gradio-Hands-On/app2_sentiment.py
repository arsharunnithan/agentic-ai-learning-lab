import gradio as gr
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

def analyze(text):
    result = sentiment(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 1)
    return f"{label} — {score}% confident"

demo = gr.Interface(
    fn=analyze,
    inputs="text",
    outputs="text",
    title="Sentiment Analyzer",
    description="Type any sentence and the AI will tell you if it's positive or negative."
)

demo.launch()
