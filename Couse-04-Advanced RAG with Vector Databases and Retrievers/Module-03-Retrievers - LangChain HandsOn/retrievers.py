from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Step 1 — Setup LLM
llm = ChatGroq(
    api_key="YOUR groq KEY",
    model_name="llama-3.1-8b-instant"
)

# Step 2 — Setup embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3 — Sample documents about a company
documents = [
    Document(page_content="Employees must use company email for all official communications. Personal email is strictly prohibited for work matters.", metadata={"topic": "email", "department": "HR"}),
    Document(page_content="The company provides health insurance coverage for all full-time employees and their immediate family members.", metadata={"topic": "benefits", "department": "HR"}),
    Document(page_content="Remote work is allowed up to 3 days per week. Employees must be available during core hours 10am to 3pm.", metadata={"topic": "remote work", "department": "HR"}),
    Document(page_content="Annual performance reviews are conducted every December. Bonuses are based on performance ratings.", metadata={"topic": "performance", "department": "HR"}),
    Document(page_content="All employees must complete cybersecurity training within 30 days of joining.", metadata={"topic": "security", "department": "IT"}),
    Document(page_content="The IT helpdesk is available Monday to Friday 9am to 6pm for technical support.", metadata={"topic": "IT support", "department": "IT"}),
    Document(page_content="Employees are entitled to 20 days of paid leave per year plus public holidays.", metadata={"topic": "leave", "department": "HR"}),
    Document(page_content="Smoking is strictly prohibited inside office premises. A designated smoking area is available outside.", metadata={"topic": "smoking", "department": "Admin"}),
]

# Step 4 — Split and store in FAISS
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)

print("✅ Vector store ready!")
print(f"Total chunks: {len(chunks)}")

print("\n" + "="*50)
print("RETRIEVER 1 — Vector Store Retriever")
print("="*50)

# Basic similarity search retriever
retriever1 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Search
query = "what is the email policy?"
docs = retriever1.invoke(query)

print(f"\nQuery: '{query}'")
print(f"Results found: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"Result {i+1}: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")

print("\n" + "="*50)
print("RETRIEVER 2 — Multi-Query Retriever")
print("="*50)

from langchain_classic.retrievers import MultiQueryRetriever

# Multi-query retriever uses LLM to generate multiple query versions
retriever2 = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

query = "what is the email policy?"
docs = retriever2.invoke(query)

print(f"\nQuery: '{query}'")
print(f"Results found: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"Result {i+1}: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")


print("\n" + "="*50)
print("RETRIEVER 3 — Self-Query Retriever")
print("="*50)

from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo

# Describe the metadata fields to the LLM
metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The topic of the document e.g. email, benefits, remote work, performance, security, IT support, leave, smoking",
        type="string"
    ),
    AttributeInfo(
        name="department",
        description="The department the policy belongs to e.g. HR, IT, Admin",
        type="string"
    )
]
# Self-Query needs Chroma instead of FAISS
chroma_vectorstore = Chroma.from_documents(chunks, embeddings)

# Create self-query retriever
retriever3 = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=chroma_vectorstore,
    document_contents="Company HR and IT policies",
    metadata_field_info=metadata_field_info,
    verbose=True,
    fix_invalid=True
)

query = "show me only IT department policies"
docs = retriever3.invoke(query)

print(f"\nQuery: '{query}'")
print(f"Results found: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"Result {i+1}: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")

print("\n" + "="*50)
print("RETRIEVER 4 — Parent Document Retriever")
print("="*50)

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryBaseStore as InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small chunks for searching (more precise embeddings)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# Large chunks to return (more context)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

# InMemoryStore stores the large parent chunks
store = InMemoryStore()

# New vectorstore for parent document retriever
parent_vectorstore = FAISS.from_documents([], embeddings) if False else FAISS.from_texts(["placeholder"], embeddings)

retriever4 = ParentDocumentRetriever(
    vectorstore=parent_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Add documents
retriever4.add_documents(documents)

query = "what is the email policy?"
docs = retriever4.invoke(query)

print(f"\nQuery: '{query}'")
print(f"Results found: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"Result {i+1}: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")



    import gradio as gr

def search(query, retriever_type, department_filter):
    if retriever_type == "Vector Store Retriever":
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(query)
    
    elif retriever_type == "Multi-Query Retriever":
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            llm=llm
        )
        docs = retriever.invoke(query)
    
    elif retriever_type == "Self-Query Retriever":
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=chroma_vectorstore,
            document_contents="Company HR and IT policies",
            metadata_field_info=metadata_field_info,
            verbose=False,
            fix_invalid=True
        )
        docs = retriever.invoke(query)
    
    elif retriever_type == "Parent Document Retriever":
        docs = retriever4.invoke(query)
    
    if not docs:
        return "No results found!"
    
    output = f"🔍 Results using {retriever_type}\n"
    output += f"Query: '{query}'\n"
    output += f"Results found: {len(docs)}\n\n"
    
    for i, doc in enumerate(docs):
        output += f"Result {i+1}:\n"
        output += f"{doc.page_content}\n"
        output += f"Topic: {doc.metadata.get('topic')} | Dept: {doc.metadata.get('department')}\n\n"
    
    return output

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 LangChain Retrievers Comparison")
    gr.Markdown("Compare how different retrievers handle the same query!")

    with gr.Row():
        query_input = gr.Textbox(
            label="Enter your query",
            placeholder="e.g. what is the email policy?"
        )
        retriever_choice = gr.Dropdown(
            choices=[
                "Vector Store Retriever",
                "Multi-Query Retriever",
                "Self-Query Retriever",
                "Parent Document Retriever"
            ],
            label="Pick a Retriever",
            value="Vector Store Retriever"
        )

    department_filter = gr.Dropdown(
        choices=["Any", "HR", "IT", "Admin"],
        label="Department Filter (for Self-Query)",
        value="Any"
    )

    search_btn = gr.Button("🔍 Search", variant="primary")
    output = gr.Textbox(label="Results", lines=15)

    search_btn.click(
        fn=search,
        inputs=[query_input, retriever_choice, department_filter],
        outputs=output
    )

demo.launch()
