import streamlit as st
import tempfile

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("📄 Chat with your PDF (RAG AI Agent)")
st.write("Upload a PDF and ask questions about it.")

# =========================
# Session State
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# =========================
# Process PDF
# =========================
if uploaded_file is not None and st.session_state.vectorstore is None:

    with st.spinner("Processing PDF..."):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Embeddings (LOCAL)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector store
        vectorstore = FAISS.from_documents(docs, embeddings)

        st.session_state.vectorstore = vectorstore

    st.success("✅ PDF processed successfully!")

# =========================
# Chat Section
# =========================
if st.session_state.vectorstore is not None:

    query = st.text_input("Ask a question about your PDF:")

    if query:
        with st.spinner("Thinking..."):

            # Retriever (LIMIT results)
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 2}
            )

            relevant_docs = retriever.invoke(query)

            # Build context (LIMIT SIZE)
            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )
            context = context[:2000]  # important for small models

            # Local LLM
            llm = OllamaLLM(model="phi")

            # 🔥 STRICT PROMPT (hallucination control)
            prompt = f"""
You MUST answer using ONLY the context provided below.

STRICT RULES:
- Do NOT say you don't have access to the document
- Do NOT use outside knowledge
- Do NOT create stories or extra explanations
- If answer is not present, say: "Not found in document"

Context:
{context}

Question:
{query}

Answer (based only on context):
"""

            response = llm.invoke(prompt)

            # =========================
            # Display Answer
            # =========================
            st.subheader("Answer")
            st.write(response.strip())

            # =========================
            # Source Context
            # =========================
            with st.expander("📌 Source Context"):
                for i, doc in enumerate(relevant_docs):
                    page = doc.metadata.get("page", "N/A")
                    snippet = doc.page_content[:200].replace("\n", " ")

                    st.write(f"**[{i+1}] Page {page}**")
                    st.write(snippet + "...")
                    st.write("---")