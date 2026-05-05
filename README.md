# 📄 Chat with PDF

An AI-powered app that lets you upload any PDF and ask questions about it using RAG (Retrieval Augmented Generation).

## 🔧 Tech Stack
- LangChain
- Groq API (LLaMA 3.3 70B)
- FAISS Vector Store
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Streamlit

## 🚀 How It Works
1. Upload any PDF file
2. PDF is split into chunks and converted to embeddings
3. Stored in FAISS vector database
4. Your question is matched to relevant chunks
5. Groq LLM generates answer from retrieved context

## ⚙️ Setup & Run

```bash
git clone <your-repo-url>
cd chat-with-pdf
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
