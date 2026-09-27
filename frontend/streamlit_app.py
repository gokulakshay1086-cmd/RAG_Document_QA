"""
Streamlit UI for the RAG Document Intelligence & Q&A System.

Run with:
    streamlit run frontend/streamlit_app.py

Expects the FastAPI backend to be running (default http://localhost:8000).
"""
import os
import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Intelligence & Q&A")
st.caption("Upload documents, then ask questions answered using Retrieval-Augmented Generation.")

# ---------------- Sidebar: health + document management ----------------

with st.sidebar:
    st.header("System status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("Backend connected")
        st.write(f"**Mode:** {health['llm_mode']}")
        st.write(f"**Documents indexed:** {health['documents_indexed']}")
        st.write(f"**Total chunks:** {health['total_chunks']}")
    except requests.exceptions.RequestException:
        st.error(f"Cannot reach backend at {API_URL}. Is it running?")

    st.divider()
    st.header("Upload a document")
    uploaded_file = st.file_uploader("PDF, DOCX, TXT, or MD", type=["pdf", "docx", "txt", "md"])
    if uploaded_file is not None and st.button("Ingest document", use_container_width=True):
        with st.spinner("Chunking, embedding, and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)
                if resp.ok:
                    data = resp.json()
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}")
                else:
                    st.error(resp.json().get("detail", "Upload failed."))
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    st.divider()
    st.header("Indexed documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=5).json()
        if not docs:
            st.info("No documents indexed yet.")
        for d in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {d['filename']}  \n`{d['chunks']} chunks`")
            if col2.button("🗑️", key=f"del-{d['document_id']}"):
                requests.delete(f"{API_URL}/documents/{d['document_id']}", timeout=10)
                st.rerun()
    except requests.exceptions.RequestException:
        pass

# ---------------- Main: Q&A ----------------

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about your documents:", placeholder="e.g. What are the key findings?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving relevant passages and generating an answer..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
            if resp.ok:
                data = resp.json()
                st.session_state.history.insert(0, {"question": question, **data})
            else:
                st.error(resp.json().get("detail", "Query failed."))
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

for item in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**Q: {item['question']}**")
        st.write(item["answer"])
        st.caption(f"Answer mode: {item['mode']}")
        if item["sources"]:
            with st.expander(f"View {len(item['sources'])} source passage(s)"):
                for s in item["sources"]:
                    st.markdown(f"**{s['filename']}** — chunk {s['chunk_index']} (score {s['score']:.2f})")
                    st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
                    st.divider()
