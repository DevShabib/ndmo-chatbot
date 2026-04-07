import os
import streamlit as st
from pathlib import Path

from groq import Groq
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="NDMO Policies Assistant",
    page_icon="📋",
    layout="centered"
)

st.title("📋 NDMO Policies Assistant")
st.caption("Ask any question about the NDMO Data Management and Personal Data Protection Standards.")

# -------------------------------
# GROQ CLIENT
# On Streamlit Cloud, secrets are stored in App Settings → Secrets
# -------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = """You are a helpful assistant specializing in the NDMO (National Data Management Office)
Data Management and Personal Data Protection Standards document.

Answer questions clearly and accurately based only on the provided context from the document.
If the answer is not found in the context, reply: "I don't have enough information in the document to answer that."
Always be concise and professional."""

# -------------------------------
# VECTORSTORE — cached so it only builds once per session
# -------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR = "./chroma_db"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

@st.cache_resource(show_spinner="📚 Loading document... (first run only, ~1 min)")
def load_vectorstore():
    pdf_path = Path("Policies.pdf")
    if not pdf_path.exists():
        st.error("Policies.pdf not found! Make sure it is in your GitHub repo root.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(CHROMA_DB_DIR).exists():
        return Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )

    try:
        loader = PDFPlumberLoader(str(pdf_path))
        docs = loader.load()
    except Exception:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

    for d in docs:
        d.metadata["source"] = pdf_path.name
        d.page_content = " ".join(d.page_content.split())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs_split = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        docs_split,
        embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

# -------------------------------
# CHAT HISTORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# CHAT INPUT
# -------------------------------
if prompt := st.chat_input("Ask about NDMO policies..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    docs = retriever.invoke(prompt)
    context = "\n\n".join([d.page_content for d in docs])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context from document:\n{context}\n\nQuestion: {prompt}"
                    }
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            answer += "\n\n📄 *Source: Policies.pdf*"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
