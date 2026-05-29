# Course 05 — Module 07: AI Nutrition Coach (Flask + Vision AI)

A production-grade Flask web application that analyzes food images
and provides instant calorie counts and personalized dietary advice
using Llama 4 Vision.

## What it does
- Upload any food photo
- Llama 4 Scout Vision identifies all food items
- Estimates calories using nutritional database reference
- Provides macronutrient breakdown (protein, carbs, fat)
- Gives personalized dietary advice
- Suggests healthier alternatives

## How to run
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`

## Tech Stack
- Flask — Python web framework (backend)
- HTML/CSS/JavaScript — custom frontend
- Groq API — fast LLM inference
- Llama 4 Scout Vision — multimodal food analysis
- Nutritional database — 15 common foods with macros

## Project Structure
```
├── app.py              ← Flask backend + Groq Vision API
├── templates/
│   └── index.html      ← Frontend (HTML/CSS/JS)
└── requirements.txt
```

## Key Concepts Demonstrated
- Flask web application development
- REST API with POST endpoint (`/analyze`)
- Base64 image encoding for Vision API
- Nutritional database as RAG context
- Frontend-backend communication with fetch API
- Markdown rendering in HTML

## Flask vs Gradio
| | Gradio | Flask |
|--|--------|-------|
| UI Control | Limited | Full HTML/CSS |
| Use case | AI demos | Real web apps |
| Port | 7860 | 5000 |
| Frontend | Auto-generated | Custom |

## Note
Add your Groq API key in app.py before running.
Never commit real API keys to GitHub.
