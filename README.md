# 🩺 Medical Chatbot – Local AI-Powered Health Q&A Assistant

This project is a locally hosted **medical chatbot** that answers a wide range of health-related queries using information from preloaded medical documents. It combines retrieval-augmented generation (RAG) with open-source tools like **LangChain**, **Ollama**, **FAISS**, and **Streamlit** to deliver private, fast, and context-aware answers — all offline.

---

## 🔍 Features

- 💬 Chat interface for asking health-related queries
- 🧠 Powered by **Mistral 7B** LLM running locally via Ollama
- 🔎 Retrieves relevant information using **FAISS vector store**
- 📄 Medical PDFs preloaded and processed for context
- 📌 Responses are strictly based on document content
- ⚡ Caching improves speed for repeated usage
- 🔐 Entirely offline — no internet or API needed

---

## 🧰 Tech Stack

| Component            | Tool / Library                                 |
|---------------------|------------------------------------------------|
| LLM Backend          | `mistral:7b` via Ollama                        |
| Document Loading     | `DirectoryLoader`, `PyPDFLoader`              |
| Text Chunking        | `RecursiveCharacterTextSplitter`              |
| Embedding Model      | `all-MiniLM-L6-v2` (HuggingFace)              |
| Vector Store         | FAISS                                          |
| Retrieval Logic      | LangChain’s `RetrievalQA`                     |
| Chat Interface       | Streamlit                                     |

## 📸 Screenshot

