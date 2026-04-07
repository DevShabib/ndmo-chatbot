import os
import streamlit as st
from groq import Groq
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===========================
# CONFIG
# ===========================
PDF_PATH = "Policies.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an expert assistant for the National Data Management Office (NDMO) policies.
Answer questions based strictly on the provided policy context.
If the answer is not in the context, say: "This information is not covered in the NDMO policy document."
Be concise, clear, and professional."""

# ===========================
# LOAD & INDEX PDF (cached)
# ===========================
@st.cache_resource(show_spinner="Building policy index... (first load only)")
def load_vectorstore():
    # Extract text from PDF
    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)

    full_text = "\n\n".join(pages)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents([full_text])

    # Build FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ===========================
# RETRIEVE RELEVANT CHUNKS
# ===========================
def get_context(query, vectorstore, k=4):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join([r.page_content for r in results])

# ===========================
# GROQ CHAT
# ===========================
def ask_groq(user_message, context, history):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history (last 6 turns to stay within token limits)
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # Add context + current question
    messages.append({
        "role": "user",
        "content": f"Context from NDMO policy document:\n{context}\n\nQuestion: {user_message}"
    })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content

# ===========================
# STREAMLIT UI
# ===========================
st.set_page_config(page_title="NDMO Policy Chatbot", page_icon="📋", layout="centered")
st.title("📋 NDMO Policy Assistant")
st.caption("Ask anything about the National Data Management Office policies.")

# Load vector store
vectorstore = load_vectorstore()

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])

# Chat input
user_input = st.chat_input("Ask about NDMO policies...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            context = get_context(user_input, vectorstore)
            answer = ask_groq(user_input, context, st.session_state.history)
        st.write(answer)

    st.session_state.history.append({"user": user_input, "assistant": answer})
