import os
import json
import streamlit as st

# ---- Configuration & Directories ----
TRANSFER_DIR = "/tmp/transfers"
TEXT_DB_FILE = "shared_texts.json"
TEXT_DB_PATH = os.path.join(TRANSFER_DIR, TEXT_DB_FILE)

os.makedirs(TRANSFER_DIR, exist_ok=True)

# ---- Helper Functions for Data Storage ----
def load_shared_texts():
    """Reads shared text snippets from the JSON database file."""
    if os.path.exists(TEXT_DB_PATH):
        try:
            with open(TEXT_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_shared_text(content):
    """Appends a new text snippet to the JSON database file."""
    texts = load_shared_texts()
    if content.strip():
        texts.append(content)
        with open(TEXT_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)

# ---- Sidebar Layout (File Transfer) ----
with st.sidebar:
    st.header("📁 File transfer")
    
    # ---- File Upload ----
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
        st.rerun()
        
    st.divider()
    
    # ---- File Download ----
    all_files = sorted(os.listdir(TRANSFER_DIR))
    downloadable_files = [f for f in all_files if f != TEXT_DB_FILE]
    
    if downloadable_files:
        st.caption("Available to download:")
        for name in downloadable_files:
            file_path = os.path.join(TRANSFER_DIR, name)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {name}", 
                        data=f.read(), 
                        file_name=name, 
                        key=f"file_dl_{name}"
                    )
    else:
        st.caption("No files yet.")

# ==========================================
# MAIN APP BODY: PUBLIC TEXT BOARD
# ==========================================
st.header("📝 Shared Text Board")
st.caption("💡 Press **Ctrl + Enter** (or Cmd + Enter) inside the box to post instantly!")

# Initializing session key to reset input area on submit
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Input field directly responds to keyboard submission shortcut
pasted = st.text_area(
    label="Paste text here:", 
    value=st.session_state.text_input,
    placeholder="Type or paste something... formatting will be preserved.",
    height=150,
    key="input_box"
)

# Broadcast button acts as a fallback or visual click indicator
if st.button("Share with Everyone") or (pasted and pasted != st.session_state.text_input):
    if pasted.strip():
        save_shared_text(pasted)
        st.success("Successfully posted to the board!")
        st.session_state.text_input = "" # Resets field data
        st.rerun()

st.divider()

# ---- Displaying Live Shared Content ----
st.subheader("📋 Live Board Feed")
shared_snippets = load_shared_texts()

if shared_snippets:
    for index, text_content in enumerate(reversed(shared_snippets)):
        with st.container(border=True):
            st.text(text_content)
            
            # Formatted Columns for Download and Copy utilities 
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.download_button(
                    label="⬇️ Download",
                    data=text_content,
                    file_name=f"shared_text_{index + 1}.txt",
                    mime="text/plain",
                    key=f"text_dl_{index}"
                )
            
            with col2:
                # Direct HTML injection creating a functional browser clipboard utility
                escaped_text = json.dumps(text_content) # Escapes quotes / newlines safely for Javascript
                button_html = f"""
                <button onclick='navigator.clipboard.writeText({escaped_text}); this.innerText="📋 Copied!"; setTimeout(() => this.innerText="📋 Copy", 2000);' 
                style='
                    background-color: transparent; 
                    color: rgb(250, 250, 250); 
                    border: 1px solid rgba(250, 250, 250, 0.2); 
                    padding: 0.4rem 0.75rem; 
                    border-radius: 0.5rem; 
                    cursor: pointer; 
                    font-size: 14px;
                    line-height: 1.6;
                '>📋 Copy</button>
                """
                st.components.v1.html(button_html, height=45)
else:
    st.info("The board is currently empty. Be the first to post something!")
