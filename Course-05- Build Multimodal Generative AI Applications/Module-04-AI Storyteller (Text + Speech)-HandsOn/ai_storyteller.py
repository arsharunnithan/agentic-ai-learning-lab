import gradio as gr
from groq import Groq
from gtts import gTTS
import tempfile
import os

client = Groq(api_key="YOUR GROQ KEY")

def generate_story_and_audio(topic):
    if not topic.strip():
        return "Please enter a topic!", None
    
    # Step 1 — Generate story using Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a creative storyteller. Generate engaging, informative and captivating stories. Keep stories between 150-200 words — long enough to be interesting but short enough for audio."
            },
            {
                "role": "user",
                "content": f"Write a creative and informative story about: {topic}"
            }
        ]
    )
    
    story = response.choices[0].message.content
    
    # Step 2 — Convert story to audio using gTTS
    tts = gTTS(text=story, lang='en', slow=False)
    
    # Step 3 — Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    
    return story, temp_file.name

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📖 AI Storyteller")
    gr.Markdown("Type any topic and get a creative story read aloud!")
    
    topic_input = gr.Textbox(
        label="Enter a topic",
        placeholder="e.g. the life span of trees, black holes, ancient Egypt..."
    )
    
    generate_btn = gr.Button("✨ Generate Story", variant="primary")
    
    story_output = gr.Textbox(
        label="Generated Story",
        lines=10
    )
    
    audio_output = gr.Audio(
        label="Listen to the Story",
        type="filepath"
    )
    
    generate_btn.click(
        fn=generate_story_and_audio,
        inputs=topic_input,
        outputs=[story_output, audio_output]
    )

demo.launch()
