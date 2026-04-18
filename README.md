# 🤖 AI RAG PDF Agent

An end-to-end Retrieval-Augmented Generation (RAG) based AI application that allows users to query PDF documents using natural language. Built with LangChain, FAISS, and a fully local LLM (Phi via Ollama), with an interactive Streamlit UI.

---

## 🚀 Features

* 📄 Upload and query any PDF document
* 🧠 RAG pipeline (Retrieval + Generation)
* 🔍 Semantic search using FAISS vector database
* ⚡ Local embeddings using MiniLM (no API cost)
* 🧩 Local LLM (Phi via Ollama) — fully offline
* 🎯 Controlled prompting to reduce hallucinations
* 📌 Source context display for explainability
* 💻 Streamlit-based interactive UI

---

## 🧠 How It Works

1. **PDF Ingestion**
   The uploaded PDF is loaded and parsed into text.

2. **Chunking**
   Text is split into smaller chunks for efficient processing.

3. **Embeddings**
   Each chunk is converted into vector embeddings using MiniLM.

4. **Vector Storage (FAISS)**
   Embeddings are stored for fast similarity search.

5. **Query Processing**

   * User query → relevant chunks retrieved
   * Context passed to LLM
   * LLM generates answer strictly based on context

---

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **FAISS (Vector DB)**
* **HuggingFace Embeddings (MiniLM)**
* **Ollama (Phi LLM)**
* **Streamlit (UI)**

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ai-rag-pdf-agent.git
cd ai-rag-pdf-agent
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python -m streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
ai-rag-pdf-agent/
│
├── streamlit_app.py      # Main UI application
├── app.py                # CLI version (for testing/debugging)
├── data/
│   └── sample.pdf        # Example PDF
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎯 Example Use Cases

* Resume analysis
* Job description understanding
* Document Q&A
* Research paper summarization
* Business document insights

---

## ⚠️ Limitations

* Small local LLM (Phi) may produce less detailed answers
* Performance depends on system RAM/CPU
* Large PDFs may require optimization

---

## 🔮 Future Improvements

* Add chat memory
* Support multiple PDFs
* Improve UI/UX
* Deploy on cloud
* Use advanced LLMs for better accuracy

---

## 🧠 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector similarity search
* Prompt engineering
* Hallucination control
* Local AI systems (no external APIs)

---

## 👨‍💻 Author

**Swastik Singh**

---

## ⭐ If you found this useful

Give this repo a star ⭐ and feel free to connect!

---
