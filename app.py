from dotenv import load_dotenv
import os

load_dotenv()

# Loaders & splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM
from langchain_ollama import OllamaLLM

# =========================
# 1. Load PDF
# =========================
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

print("Total pages:", len(documents))

# =========================
# 2. Split into chunks
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print("Total chunks:", len(docs))

# =========================
# 3. Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# =========================
# 4. Vector Store
# =========================
vectorstore = FAISS.from_documents(docs, embeddings)

# =========================
# 5. Retriever
# =========================
retriever = vectorstore.as_retriever()

# =========================
# 6. LLM (Local)
# =========================
llm = OllamaLLM(model="phi")

# =========================
# 7. Chat Loop
# =========================
print("\n✅ Chat with your PDF (type 'exit' to quit)\n")

while True:
    query = input("Ask something: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant docs
    relevant_docs = retriever.invoke(query)

    # Limit context (important)
    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])

    # Controlled prompt
    prompt = f"""
You are an AI assistant. Answer ONLY using the provided context.

Rules:
- Do NOT add extra information
- Do NOT create stories or unrelated content
- If answer is not in context, say "Not found in document"

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate response
    response = llm.invoke(prompt)

    print("\nAnswer:", response.strip())

    # =========================
    # Source Context (Explainability)
    # =========================
    print("\n--- Source Context ---")

    for i, doc in enumerate(relevant_docs[:2]):
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:200].replace("\n", " ")

        print(f"\n[{i+1}] Page: {page}")
        print(f"Snippet: {snippet}...")

    print("\n" + "="*50 + "\n")