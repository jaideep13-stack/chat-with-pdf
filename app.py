import streamlit as st
import tempfile, os

st.title("📄 Chat with PDF")

groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

if not uploaded_file:
    st.info("Please upload a PDF file in the sidebar.")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

with st.spinner("Reading PDF..."):
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

st.success("PDF loaded! Ask your question below.")
question = st.text_input("Ask something about the PDF:")

if question:
    with st.spinner("Thinking..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        response = llm.invoke(prompt)
        st.write(response.content)

os.unlink(tmp_path)