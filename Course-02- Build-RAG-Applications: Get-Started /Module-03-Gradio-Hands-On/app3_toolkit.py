import gradio as gr
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
generator = pipeline("text-generation", model="gpt2")

def analyze(text, task):
    if task == "Sentiment Analysis":
        result = sentiment(text)[0]
        label = result['label']
        score = round(result['score'] * 100, 1)
        return f"{label} — {score}% confident"
    elif task == "Continue my text (GPT-2)":
        result = generator(text, max_new_tokens=50, num_return_sequences=1)[0]
        return result['generated_text']

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="Enter your text here", lines=5),
        gr.Dropdown(
            choices=["Sentiment Analysis", "Continue my text (GPT-2)"],
            label="Pick a task",
            value="Sentiment Analysis"
        )
    ],
    outputs=gr.Textbox(label="Result"),
    title="My AI Toolkit",
    description="Pick a task and let the AI do the work."
)

demo.launch()
