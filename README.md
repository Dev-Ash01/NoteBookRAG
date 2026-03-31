
# 📘 NoteBookRAG

A modular **Retrieval-Augmented Generation (RAG)** application that allows users to organize documents into notebooks and interact with them through a conversational AI interface.

---

## 🚀 Overview

**NoteBookRAG** is designed to solve the problem of interacting with large collections of documents efficiently. It enables users to:

* Organize documents into **separate notebooks**
* Automatically process and embed files
* Query documents using **natural language**
* Switch between different **LLM providers**

This makes it ideal for **developers, researchers, and knowledge workers** dealing with multiple document contexts.

---

## ✨ Features

* 📂 **Multi-Notebook System**
  Create and manage independent notebooks for different projects.

* 📄 **Document Upload Support**
  Supports PDF, TXT, and Markdown files.

* ⚙️ **Automatic Document Processing**

  * Text extraction
  * Chunking
  * Embedding generation

* 🧠 **RAG-based Chat Interface**
  Ask questions and get contextual answers from your documents.

* 🔌 **Multi-LLM Support**

  * Groq (cloud-based)
  * Ollama (local models)

* 💾 **Persistent Storage**

  * ChromaDB (vector storage)
  * SQLite (metadata)
  * Organized file system

* 🔄 **Smart Processing**
  Avoids re-processing already embedded documents.

---

## 🏗️ Architecture

The project follows a **modular architecture**, making it scalable and easy to maintain.

### Core Components

1. **Document Processing**

   * Extracts and splits text
   * Generates embeddings using transformer models

2. **Vector Store (ChromaDB)**

   * Stores embeddings
   * Enables semantic search

3. **Conversation Manager**

   * Handles LLM interactions
   * Builds prompts and responses

4. **Database (SQLite)**

   * Stores notebook and file metadata
   * Tracks processing status

5. **File System Manager**

   * Organizes uploaded files per notebook

---

## 📁 Project Structure

```
NotebookRAG/
├── backend/
│   ├── main.py          # FastAPI app entry point
│   ├── ingest.py        # Document parsing & chunking
│   ├── embedder.py      # Embedding logic
│   ├── retriever.py     # Vector search
│   ├── synthesizer.py   # LLM call
│   └── vector_store/    # Where FAISS index lives
├── frontend/
│   └── index.html       # Simple query UI
├── documents/           # User uploads go here
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

* Python 3.10+
* Git (optional)
* API key (for Groq or other providers)

---

### Steps

```bash
# Clone repository
git clone https://github.com/Dev-Ash01/NoteBookRAG.git
cd NoteBookRAG

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

(Optional for GPU support)

```bash
pip install -r requirements_cuda.txt
```

---

### 🔑 Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

App will be available at:

```
http://localhost:8501
```

---

## 🧑‍💻 Usage Guide

### 1. Create Notebook

* Enter notebook name in sidebar
* Click **Create Notebook**

### 2. Upload Documents

* Upload PDF / TXT / MD files
* Files stored per notebook

### 3. Process Documents

* Click **Process Files**
* System generates embeddings

### 4. Select LLM

* Choose provider (Groq / Ollama)
* Select model

### 5. Chat with Documents

* Ask questions
* Get context-aware responses

---

## 🧠 How It Works (RAG Flow)

1. User asks a question
2. Relevant chunks retrieved from vector DB
3. Context injected into prompt
4. LLM generates response

---

## 📌 Use Cases

* 📚 Research paper analysis
* 🧑‍💻 Code/documentation assistant
* 📊 Knowledge base querying
* 🏢 Enterprise document search

---

## 🔮 Future Improvements

* Support for images & multimedia
* Document version control
* Better UI/UX enhancements
* Advanced retrieval optimization

