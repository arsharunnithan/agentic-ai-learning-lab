import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1 — Load the LLM
llm = ChatGroq(
    api_key="YOUR_GROQ_KEY_HERE",
    model_name="llama-3.1-8b-instant"
)

qa_chain = None

def build_qa_chain(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    chunks = splitter.split_documents(documents)

    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build RAG chain using modern LangChain syntax
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

If the answer is not in the context, say "I don't find that in the document."
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def upload_pdf(pdf_file):
    global qa_chain
    qa_chain = build_qa_chain(pdf_file)
    return "✅ PDF loaded! Ask me anything about it."

def answer_question(question):
    if qa_chain is None:
        return "Please upload a PDF first!"
    result = qa_chain.invoke(question)
    return result

with gr.Blocks() as demo:
    gr.Markdown("# 📄 QA Bot — Chat with your Document")
    gr.Markdown("Upload a PDF, then ask questions about it.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Status")

    pdf_input.change(fn=upload_pdf, inputs=pdf_input, outputs=upload_status)

    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer", lines=5)
    ask_btn = gr.Button("Ask")

    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

demo.launch()
