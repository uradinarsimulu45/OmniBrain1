"""Streamlit Community Cloud interface for the OmniBrain PDF RAG assistant."""

import os
import tempfile
from pathlib import Path

import streamlit as st

from rag_chat import ConversationalRAGPipeline


st.set_page_config(
    page_title="OmniBrain | Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 OmniBrain")
st.caption("Multi-document AI assistant · Ask questions grounded in your PDF files")

# Streamlit Cloud secrets are exposed through st.secrets. Environment variables
# are also supported for local development.
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document workspace")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF files to create a searchable knowledge base.",
    )

    if st.button("Process documents", type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        temp_paths = []
        try:
            with st.spinner("Extracting text, creating embeddings and building the FAISS index..."):
                with tempfile.TemporaryDirectory(prefix="omnibrain_") as temp_dir:
                    for uploaded in uploaded_files:
                        safe_name = Path(uploaded.name).name
                        file_path = Path(temp_dir) / safe_name
                        file_path.write_bytes(uploaded.getvalue())
                        temp_paths.append(str(file_path))

                    pipeline = ConversationalRAGPipeline(
                        model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                    )
                    metadata = pipeline.load_documents(temp_paths)

                    if GEMINI_API_KEY:
                        pipeline.init_client(api_key=GEMINI_API_KEY)

                    st.session_state.rag_pipeline = pipeline
                    st.session_state.documents = metadata
                    st.session_state.messages = []

            st.success(f"Indexed {len(metadata)} document(s) successfully.")
        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

    if st.session_state.documents:
        st.subheader("Indexed documents")
        for doc in st.session_state.documents:
            st.write(f"📄 **{doc.get('filename', 'PDF')}**")
            st.caption(
                f"{doc.get('page_count', '?')} pages · "
                f"{doc.get('chunk_count', '?')} text chunks"
            )
        if st.button("Clear workspace", use_container_width=True):
            st.session_state.rag_pipeline = None
            st.session_state.documents = []
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Retrieval: Sentence Transformers + FAISS")
    st.caption("Answer generation: Google Gemini")
    if not GEMINI_API_KEY:
        st.warning("Add GEMINI_API_KEY in Streamlit Cloud → App settings → Secrets to enable AI answers.")

if st.session_state.rag_pipeline is None:
    st.info("Upload one or more PDF files in the sidebar, then select **Process documents** to get started.")
else:
    st.subheader("Chat with your documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source.get('filename', source.get('source', 'Document'))}** "
                            f"· Page {source.get('page', 'N/A')} "
                            f"· {source.get('confidence', 'N/A')} confidence "
                            f"· Score {source.get('score', 0):.3f}"
                        )
                        st.caption(source.get("text_preview", ""))

    prompt = st.chat_input("Ask a question about your uploaded documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if not GEMINI_API_KEY:
                    st.error("Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Cloud Secrets and reboot the app.")
                else:
                    with st.spinner("Searching documents and generating an answer..."):
                        response = st.session_state.rag_pipeline.chat(prompt)
                    st.markdown(response["answer"])
                    if response.get("sources"):
                        with st.expander("View retrieved sources"):
                            for source in response["sources"]:
                                st.markdown(
                                    f"**{source.get('filename', source.get('source', 'Document'))}** "
                                    f"· Page {source.get('page', 'N/A')} "
                                    f"· {source.get('confidence', 'N/A')} confidence "
                                    f"· Score {source.get('score', 0):.3f}"
                                )
                                st.caption(source.get("text_preview", ""))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                    })
            except Exception as exc:
                st.error(f"Unable to generate an answer: {exc}")
