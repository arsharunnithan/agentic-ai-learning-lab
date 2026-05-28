import gradio as gr
import whisper
from groq import Groq
import tempfile
import os

# Load Whisper model (downloads ~150MB first time)
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("✅ Whisper ready!")

# Setup Groq
client = Groq(api_key="GROQ_KEY")

def process_meeting(audio_file):
    if audio_file is None:
        return "Please upload an audio file!", ""
    
    # Step 1 — Transcribe audio with Whisper
    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_file)
    transcript = result["text"]
    print(f"Transcript: {transcript[:100]}...")
    
    # Step 2 — Generate meeting minutes with Groq
    print("Generating meeting minutes...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a professional meeting assistant. 
Your job is to take raw meeting transcripts and convert them into 
clean, structured meeting minutes.

Always format your output exactly like this:

## 📋 Meeting Summary
[2-3 sentence overview]

## ✅ Key Decisions
- [decision 1]
- [decision 2]

## 📌 Action Items
- [person/team]: [task] by [deadline if mentioned]

## 🔑 Key Points Discussed
- [point 1]
- [point 2]

## 📅 Follow-up Required
- [follow-up item]"""
            },
            {
                "role": "user",
                "content": f"Please convert this meeting transcript into structured meeting minutes:\n\n{transcript}"
            }
        ]
    )
    
    minutes = response.choices[0].message.content
    return transcript, minutes

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ AI Meeting Assistant")
    gr.Markdown("Upload a meeting recording and get structured meeting minutes instantly!")
    
    with gr.Row():
        audio_input = gr.Audio(
            label="Upload Meeting Recording",
            type="filepath"
        )
    
    process_btn = gr.Button("⚡ Generate Meeting Minutes", variant="primary")
    
    with gr.Tabs():
        with gr.Tab("📋 Meeting Minutes"):
            minutes_output = gr.Markdown(label="Structured Meeting Minutes")
        with gr.Tab("📝 Raw Transcript"):
            transcript_output = gr.Textbox(
                label="Raw Transcript",
                lines=15
            )
    
    process_btn.click(
        fn=process_meeting,
        inputs=audio_input,
        outputs=[transcript_output, minutes_output]
    )

demo.launch()
