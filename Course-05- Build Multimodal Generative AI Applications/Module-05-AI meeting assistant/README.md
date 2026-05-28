# Course 05 — Module 05: AI Meeting Assistant

An AI-powered meeting assistant that transcribes audio recordings
and generates structured meeting minutes automatically.

## Two Versions

### Version 1 — Local Whisper
`meeting_assistant_whisper.py`
- Uses OpenAI Whisper running locally on your CPU
- No additional API needed
- Slower (~70 seconds for short audio)
- Works offline

### Version 2 — Groq Whisper API
`meeting_assistant_groq.py`
- Uses Whisper Large V3 via Groq's servers
- Much faster (~5 seconds)
- Requires Groq API key
- Recommended for real use

## What it does
- Upload any MP3/WAV meeting recording
- Whisper transcribes speech to text
- Groq + Llama 3.1 structures transcript into:
  - Meeting Summary
  - Key Decisions
  - Action Items
  - Key Points Discussed
  - Follow-up Required

## How to run
pip install -r requirements.txt

# Local version
python meeting_assistant_whisper.py

# Groq API version (recommended)
python meeting_assistant_groq.py

## Tech Stack
- Gradio — web interface with tabs
- OpenAI Whisper — speech to text (local)
- Groq Whisper API — speech to text (cloud)
- Groq + Llama 3.1 — meeting minutes generation

## How it works
Audio file (MP3/WAV)
↓
Whisper → raw transcript text
↓
Groq + Llama 3.1 → structured meeting minutes
↓
Gradio displays transcript + formatted minutes

## Key Concepts Learned
- Speech-to-Text (STT) with Whisper
- Local vs API-based model tradeoffs
- Audio file processing in Python
- Structured output formatting with LLMs
- Gradio Tabs for organizing output

## Local vs API Tradeoff
| | Local Whisper | Groq Whisper API |
|--|--------------|-----------------|
| Speed | ~70 seconds | ~5 seconds |
| Cost | Free | Free tier |
| Internet | Not needed | Required |
| Accuracy | Good | Better |

## Note
Add your Groq API key before running.
Never commit real API keys to GitHub.
