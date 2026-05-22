import chromadb
from chromadb.utils import embedding_functions

import gradio as gr

# Create client
client = chromadb.Client()

# Create embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="employees",
    embedding_function=ef
)

print("Collection created:", collection.name)
print("Total documents:", collection.count())

# Employee records — structured data
employees = [
    {
        "id": "emp_001",
        "name": "Alice Johnson",
        "role": "Data Scientist",
        "department": "Analytics",
        "skills": "Python, Machine Learning, TensorFlow, SQL",
        "experience": "5 years"
    },
    {
        "id": "emp_002",
        "name": "Bob Smith",
        "role": "Backend Developer",
        "department": "Engineering",
        "skills": "Java, Spring Boot, PostgreSQL, Docker",
        "experience": "7 years"
    },
    {
        "id": "emp_003",
        "name": "Carol White",
        "role": "AI Engineer",
        "department": "Research",
        "skills": "Python, PyTorch, NLP, LLMs, RAG",
        "experience": "3 years"
    },
    {
        "id": "emp_004",
        "name": "David Lee",
        "role": "Data Analyst",
        "department": "Finance",
        "skills": "Excel, SQL, Power BI, Python",
        "experience": "4 years"
    },
    {
        "id": "emp_005",
        "name": "Eva Martinez",
        "role": "ML Engineer",
        "department": "Product",
        "skills": "Python, Scikit-learn, MLflow, AWS",
        "experience": "6 years"
    }
]

print("Number of employees:", len(employees))

# Convert each employee dictionary to a text description
def employee_to_text(emp):
    return f"{emp['name']} is a {emp['role']} in the {emp['department']} department. \
Skills: {emp['skills']}. Experience: {emp['experience']}."

# Test it on one employee
print(employee_to_text(employees[0]))

# Convert ALL employees to text using a loop
documents = []
for emp in employees:
    text = employee_to_text(emp)
    documents.append(text)

print(documents)

ids = [emp['id'] for emp in employees]
print(ids)

metadatas = [emp for emp in employees]
print(metadatas)

# Add all employees to ChromaDB
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print("Total documents after adding:", collection.count())

# Search for similar employees
query = "I need someone who knows Python and machine learning"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\nQuery:", query)
print("\nTop 3 matches:")
for i, (doc, metadata, distance) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    similarity = round((1 - distance) * 100, 1)
    print(f"\nRank {i+1} — {metadata['name']} ({similarity}% match)")
    print(f"Role: {metadata['role']} | Department: {metadata['department']}")
    print(f"Skills: {metadata['skills']}")


def search_employees(query):
    if not query.strip():
        return "Please enter a search query!"
    
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    output = f"🔍 Top matches for: '{query}'\n\n"
    for i, (metadata, distance) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = round((1 - distance) * 100, 1)
        output += f"Rank {i+1} — {metadata['name']} ({similarity}% match)\n"
        output += f"Role: {metadata['role']} | Dept: {metadata['department']}\n"
        output += f"Skills: {metadata['skills']}\n"
        output += f"Experience: {metadata['experience']}\n\n"
    
    return output

with gr.Blocks() as demo:
    gr.Markdown("# 👥 Employee Similarity Search")
    gr.Markdown("Find the right employee for any job requirement!")
    
    query_input = gr.Textbox(
        label="What are you looking for?",
        placeholder="e.g. I need a Python expert with ML experience"
    )
    search_btn = gr.Button("🔍 Find Employees", variant="primary")
    output = gr.Textbox(label="Top Matches", lines=15)
    
    search_btn.click(fn=search_employees, inputs=query_input, outputs=output)

demo.launch()
