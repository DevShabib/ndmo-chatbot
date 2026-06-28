import os
import streamlit as st
from groq import Groq
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


TRANSFER_DIR = "/tmp/transfers"
os.makedirs(TRANSFER_DIR, exist_ok=True)

with st.sidebar:
    st.header("📁 File transfer")

    up = st.file_uploader("Upload")
    if up is not None:
        with open(os.path.join(TRANSFER_DIR, up.name), "wb") as f:
            f.write(up.getbuffer())
        st.success(f"Saved {up.name}")

    st.divider()
    files = sorted(os.listdir(TRANSFER_DIR))
    if files:
        st.caption("Available to download:")
        for name in files:
            with open(os.path.join(TRANSFER_DIR, name), "rb") as f:
                st.download_button(f"⬇️ {name}", f.read(), file_name=name, key=name)
    else:
        st.caption("No files yet.")
