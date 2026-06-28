import os
import streamlit as st
from groq import Groq
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
# (your other imports already there)

TRANSFER_DIR = "/tmp/transfers"
os.makedirs(TRANSFER_DIR, exist_ok=True)

with st.sidebar:
    st.header("📁 File upload")

    up = st.file_uploader("Upload a file")
    if up is not None:
        dest = os.path.join(TRANSFER_DIR, up.name)
        data = up.getbuffer()
        total = len(data)

        progress = st.progress(0, text="Starting upload...")
        chunk = 1024 * 256  # 256 KB per write
        written = 0

        with open(dest, "wb") as f:
            for i in range(0, total, chunk):
                f.write(data[i:i + chunk])
                written += len(data[i:i + chunk])
                pct = int(written / total * 100)
                progress.progress(pct, text=f"Saving... {pct}%")

        progress.progress(100, text="Done ✅")
        st.success(f"Saved {up.name} ({total / 1024 / 1024:.1f} MB)")
