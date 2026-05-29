from flask import Flask, render_template, request, jsonify
from groq import Groq
import base64
import os

app = Flask(__name__)
client = Groq(api_key="GROQ KEY")

# Nutritional database
nutrition_db = {
    "rice": {"calories": 206, "protein": 4.3, "carbs": 44.5, "fat": 0.4, "serving": "1 cup cooked"},
    "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "serving": "100g"},
    "egg": {"calories": 78, "protein": 6, "carbs": 0.6, "fat": 5, "serving": "1 large egg"},
    "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4, "serving": "1 medium"},
    "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "serving": "1 medium"},
    "bread": {"calories": 79, "protein": 2.7, "carbs": 15, "fat": 1, "serving": "1 slice"},
    "milk": {"calories": 149, "protein": 8, "carbs": 12, "fat": 8, "serving": "1 cup"},
    "pasta": {"calories": 220, "protein": 8, "carbs": 43, "fat": 1.3, "serving": "1 cup cooked"},
    "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fat": 13, "serving": "100g"},
    "broccoli": {"calories": 55, "protein": 3.7, "carbs": 11, "fat": 0.6, "serving": "1 cup"},
    "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10, "serving": "1 slice"},
    "burger": {"calories": 540, "protein": 34, "carbs": 40, "fat": 27, "serving": "1 burger"},
    "salad": {"calories": 20, "protein": 1.5, "carbs": 3.5, "fat": 0.2, "serving": "1 cup"},
    "orange": {"calories": 62, "protein": 1.2, "carbs": 15, "fat": 0.2, "serving": "1 medium"},
    "almonds": {"calories": 164, "protein": 6, "carbs": 6, "fat": 14, "serving": "1 oz (28g)"},
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Build nutrition context
    nutrition_context = "Nutritional Reference Database:\n"
    for food, data in nutrition_db.items():
        nutrition_context += f"- {food}: {data['calories']} cal, {data['protein']}g protein, {data['carbs']}g carbs, {data['fat']}g fat ({data['serving']})\n"
    
    # Send to Llama Vision
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an expert AI Nutrition Coach with deep knowledge of food and dietetics.

{nutrition_context}

Analyze this food image and provide:

1. **Food Items Identified** — list everything you see
2. **Estimated Calories** — total and per item
3. **Macronutrients** — protein, carbs, fat breakdown
4. **Nutritional Assessment** — is this meal healthy?
5. **Personalized Advice** — 3 specific dietary recommendations
6. **Healthier Alternatives** — suggest improvements if needed

Be specific, accurate and helpful. Format clearly."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    analysis = response.choices[0].message.content
    return jsonify({"analysis": analysis})

if __name__ == "__main__":
    app.run(debug=True)
