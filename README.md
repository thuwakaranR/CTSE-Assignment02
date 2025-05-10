
# 🤖 CTSE Lecture Notes Chatbot (LLM + RAG System)

A lightweight Retrieval-Augmented Generation (RAG) chatbot for querying **Current Trends in Software Engineering (CTSE)** notes using natural language. Built with **LangChain**, **ChromaDB**, and **HuggingFace’s flan-t5-base** for fast, local Q&A on academic PDFs.

---

## 📁 Project Structure

```
.
├── CTSE_LLM_Chatbot.ipynb       # Jupyter Notebook implementation
├── data/                        # Folder for CTSE lecture PDFs
├── ctse_db/                     # Vector store (auto-generated)
├── CTSE_Question_Set.txt        # Sample validation queries
└── CTSE_Assignment02_Report.pdf # Final report
```

---

## ⚙️ Setup

### Requirements

- Python 3.10+
- Jupyter Notebook
- Recommended: virtualenv or conda

### Install Dependencies

```bash
pip install langchain langchain-community langchain-huggingface chromadb transformers sentence-transformers
```

---

## 🚀 Run Instructions

1. Add your CTSE PDF to `data/` (e.g., `ctse_notes.pdf`)
2. Open and run all cells in `CTSE_LLM_Chatbot.ipynb`

---

## 💡 Features

- Semantic search over PDF-based notes
- Local inference with `flan-t5-base`
- Fast answers with page-level relevance
- Interactive Q&A via terminal input

---

## 🧪 Sample Usage

```python
qa_chain.invoke({"query": "What is software architecture?"})
```

```python
interactive_ctse_bot()  # For continuous chat
```

---

## 🔧 Tech Stack

- **LLM**: flan-t5-base (HuggingFace)
- **Embeddings**: all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **Framework**: LangChain

---

## 📌 Conclusion

A private, efficient, and modular educational chatbot to enhance self-learning through natural language interaction with course material.

> 🧑‍💻 Developed for CTSE Assignment 02 – Current Trends in Software Engineering
