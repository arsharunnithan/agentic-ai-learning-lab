import gradio as gr
from groq import Groq
import os

# Setup Groq
client = Groq(api_key="GROQ_KEY")

def process_meeting(audio_file):
    if audio_file is None:
        return "Please upload an audio file!", ""
    
    # Step 1 — Transcribe using Groq's Whisper API (fast!)
    print("Transcribing audio...")
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
        )
    transcript = transcription.text
    print(f"Transcript: {transcript[:100]}...")
    
    # Step 2 — Generate meeting minutes with Groq
    print("Generating meeting minutes...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a professional meeting assistant. 
Always format your output exactly like this:

## 📋 Meeting Summary
[2-3 sentence overview]

## ✅ Key Decisions
- [decision 1]

## 📌 Action Items
- [person/team]: [task] by [deadline if mentioned]

## 🔑 Key Points Discussed
- [point 1]

## 📅 Follow-up Required
- [follow-up item]"""
            },
            {
                "role": "user",
                "content": f"Convert this meeting transcript into structured meeting minutes:\n\n{transcript}"
            }
        ]
    )
    
    minutes = response.choices[0].message.content
    return transcript, minutes

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ AI Meeting Assistant")
    gr.Markdown("Upload a meeting recording and get structured meeting minutes instantly!")
    
    audio_input = gr.Audio(
        label="Upload Meeting Recording",
        type="filepath"
    )
    
    process_btn = gr.Button("⚡ Generate Meeting Minutes", variant="primary")
    
    with gr.Tabs():
        with gr.Tab("📋 Meeting Minutes"):
            minutes_output = gr.Markdown()
        with gr.Tab("📝 Raw Transcript"):
            transcript_output = gr.Textbox(lines=15)
    
    process_btn.click(
        fn=process_meeting,
        inputs=audio_input,
        outputs=[transcript_output, minutes_output]
    )

demo.launch()
