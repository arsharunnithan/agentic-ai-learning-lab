# Course 05 — Module 04: AI Storyteller (Text + Speech)

An AI-powered storyteller that generates creative stories on any 
topic and converts them to audio using Text-to-Speech.

## What it does
- User enters any topic
- Groq + Llama 3.1 generates a creative 150-200 word story
- gTTS converts the story to speech (MP3)
- Gradio plays the audio with a waveform visualizer

## How to run
pip install -r requirements.txt

python ai_storyteller.py

## Tech Stack
- Gradio — web interface with audio player
- Groq API — fast LLM inference
- Llama 3.1 8B — story generation
- gTTS (Google Text-to-Speech) — converts text to audio
- tempfile — handles temporary MP3 file storage

## How it works
User types topic
↓
Groq + Llama 3.1 generates creative story
↓
gTTS converts story text → MP3 audio file
↓
Gradio displays story text + plays audio

## Multimodal concepts demonstrated
- Text processing — LLM generates natural language story
- Text-to-Speech (TTS) — converts written text to spoken audio
- Multimodal output — same content in two modalities (text + audio)

## Example
Topic: "Moon"
Output: "The Lunar Dance: A Tale of the Moon's Magical Rhythm"
→ Full story about lunar phases, tides, celestial ballet
→ Read aloud with natural English voice

## Note
Add your Groq API key in ai_storyteller.py before running.
Never commit real API keys to GitHub.
