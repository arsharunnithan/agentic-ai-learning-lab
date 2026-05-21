# Introduction to Gradio

> A practical reference for building interactive web interfaces for AI and machine learning models using Gradio.

---

## Table of Contents

1. [What is Gradio?](#what-is-gradio)
2. [Why Use Gradio?](#why-use-gradio)
3. [How Gradio Works — Setup Flow](#how-gradio-works--setup-flow)
4. [Installation](#installation)
5. [The Interface Class](#the-interface-class)
6. [Examples](#examples)
   - [Simple Text Interface](#simple-text-interface)
   - [Multiple Inputs](#multiple-inputs)
   - [File Upload Interface](#file-upload-interface)
   - [Image Captioning with BLIP](#image-captioning-with-blip)
   - [Image Classification with ResNet-18](#image-classification-with-resnet-18)
7. [Common Gradio Components](#common-gradio-components)
8. [Summary](#summary)

---

## What is Gradio?

**Gradio** is an open-source Python library for creating **customizable, web-based user interfaces** for machine learning models, APIs, and any Python function.

- No JavaScript, CSS, or web hosting experience required
- Works in code editors, Jupyter Notebooks, and Google Colab
- Interfaces can be shared via **unique public URLs**
- Designed specifically for ease of use with ML models and computational tools

---

## Why Use Gradio?

| Benefit | Description |
|---|---|
| **Ease of use** | Build a full UI with just a few lines of Python |
| **Flexibility** | Supports text, images, files, numbers, sliders, and 30+ other components |
| **Sharing** | Share interfaces via unique URLs — no deployment setup needed |
| **Collaboration** | Collect feedback from non-technical users without them needing to code |

---

## How Gradio Works — Setup Flow

```
1. Write Python code
        ↓
2. Create Gradio Interface (define inputs & outputs)
        ↓
3. Launch Gradio server using .launch()
        ↓
4. Access via local URL (http://127.0.0.1:7860) or public URL
        ↓
5. Users interact in real time — inputs in, outputs out
```

---

## Installation

```bash
pip install gradio
```

```python
import gradio as gr
```

---

## The Interface Class

`gr.Interface` is the core component of Gradio. It wraps any Python function with an interactive UI.

```python
demo = gr.Interface(
    fn=your_function,       # the Python function to wrap
    inputs=[...],           # Gradio input component(s)
    outputs=[...],          # Gradio output component(s)
    title="App Title",      # optional
    description="...",      # optional
    examples=[...]          # optional — prepopulate with sample inputs
)
demo.launch(server_name="127.0.0.1", server_port=7860)
```

**Three core arguments:**

| Argument | Description |
|---|---|
| `fn` | Any Python function — simple or complex (model inference, calculators, generators) |
| `inputs` | Gradio component(s) matching the number of function arguments |
| `outputs` | Gradio component(s) matching the number of function return values |

> If your function takes multiple arguments, pass a **list** of input components. Same for multiple return values.

---

## Examples

### Simple Text Interface

```python
import gradio as gr

def echo_text(text):
    return text

demo = gr.Interface(
    fn=echo_text,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Textbox(label="Output")
)
demo.launch()
```

**Result:** A web interface with one text input box and one text output box.

---

### Multiple Inputs

```python
import gradio as gr

def combine(name, number):
    return f"Hello {name}, your number is {number}"

demo = gr.Interface(
    fn=combine,
    inputs=[gr.Textbox(label="Name"), gr.Number(label="Number")],
    outputs=gr.Textbox(label="Result")
)
demo.launch()
```

**Result:** Two input fields — one text, one numeric — feeding into the same function.

---

### File Upload Interface

```python
import gradio as gr

def count_files(files):
    return len(files)

demo = gr.Interface(
    fn=count_files,
    inputs=gr.File(file_count="multiple"),
    outputs=gr.Textbox(label="Number of files")
)
demo.launch()
```

- `gr.File` allows users to **upload or drag-and-drop files**
- Supports multiple file uploads
- Provides file paths to the backend function for processing
- Generates a **unique web interface link** usable from anywhere while the session runs

---

### Image Captioning with BLIP

Uses the BLIP (Bootstrapped Language Image Pretraining) model to generate captions from uploaded images.

```python
pip install transformers torch
```

```python
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption."
)
iface.launch(server_name="127.0.0.1", server_port=7860)
```

**Use case:** Auto-generating descriptive names for photos, helping visually impaired users understand image content, organizing large digital asset libraries.

---

### Image Classification with ResNet-18

Classifies images into 1,000 categories using a pretrained ResNet-18 model from PyTorch Hub.

**Step 1 — Load the model:**
```python
import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
```

**Step 2 — Define the prediction function:**
```python
import requests
from torchvision import transforms

# Download ImageNet class labels
response = requests.get("https://git.io/JJkYN")
labels = [l.strip() for l in response.text.split("\n") if l.strip()]

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(inp):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences
```

> **Softmax** converts raw model output logits (any real number) into probabilities that sum to 1 — making outputs interpretable as confidence levels per class.

**Step 3 — Create the Gradio interface:**
```python
import gradio as gr

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),   # shows only top 3 predictions
    examples=["lion.jpg", "cheetah.jpg"]   # replace with your own image paths
).launch()
```

**Result:** Drag-and-drop an image → see the top 3 predicted classes with confidence scores.

---

## Common Gradio Components

| Component | Usage | Type |
|---|---|---|
| `gr.Textbox()` | Text input or output field | Input / Output |
| `gr.Number()` | Numeric input field | Input |
| `gr.Slider()` | Sliding range selector | Input |
| `gr.Image()` | Image upload or display | Input / Output |
| `gr.File()` | File upload (single or multiple) | Input |
| `gr.Label()` | Classification labels with confidence scores | Output |
| `gr.HTML()` | Raw HTML output | Output |

> Gradio has **30+ built-in components** designed for ML applications.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Gradio** | Open-source Python library for building web UIs for ML models and functions |
| **`gr.Interface`** | Core class — wraps any Python function with inputs, outputs, title, description, examples |
| **`fn`** | Any Python function — simple echo, ML model inference, file counter, etc. |
| **`inputs` / `outputs`** | Lists of Gradio components matching function arguments and return values |
| **`.launch()`** | Starts local server; add `share=True` to generate a public URL |
| **`gr.Textbox`** | Text input/output field |
| **`gr.Number`** | Numeric input field |
| **`gr.File`** | File upload with multiple file support |
| **`gr.Image`** | Image upload; use `type="pil"` for PIL Image objects |
| **`gr.Label`** | Classification output with `num_top_classes` parameter |
| **Softmax** | Converts logits to probabilities summing to 1 — used in classification output |

---

*Notes based on: Introduction to Gradio — Reading + Video Lecture (Course 02 — Build RAG Applications)*
