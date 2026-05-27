import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(page_title="YouTube QA Bot", page_icon="🎥")
st.title("🎥 YouTube QA Bot")
st.markdown("Paste a YouTube URL and ask questions about the video!")

# Setup
@st.cache_resource
def load_models():
    llm = ChatGroq(
        api_key="YOUR GROQ KEY",
        model_name="llama-3.1-8b-instant"
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

llm, embeddings = load_models()

# Step 1 — URL input
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if url:
    with st.spinner("Loading transcript..."):
        try:
            # Step 2 — Load transcript
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            documents = loader.load()

            if not documents:
                st.error("No transcript found for this video!")
            else:
                # Step 3 — Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=64
                )
                chunks = splitter.split_documents(documents)
                st.success(f"✅ Loaded! Split into {len(chunks)} chunks.")

                # Step 4 — Embed and store in FAISS
                vectorstore = FAISS.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # Step 5 — Build RAG chain
                prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following transcript context:

{context}

Question: {question}

If the answer is not in the transcript, say "I don't find that in the video."
""")
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # Step 6 — Question input
                question = st.text_input("Ask a question about the video")

                if question:
                    with st.spinner("Thinking..."):
                        answer = chain.invoke(question)
                        st.markdown("### Answer:")
                        st.write(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
