import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os
from datetime import datetime


st.set_page_config(
    page_title="DocMind",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Nunito:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp { background: #f0e8d8; color: #1c1408; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 860px; }

.hero-wrap {
    background: linear-gradient(135deg, #2c1a08 0%, #3d2510 100%);
    border: 2px solid #5a3820; border-radius: 20px;
    padding: 2rem 2.2rem 1.6rem; margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px #1c140840;
}
.hero-wrap h1 {
    font-family: 'Lora', serif; font-size: 2.1rem; font-weight: 600;
    color: #f5e6c8; margin: 0 0 6px 0; line-height: 1.2;
}
.hero-wrap .tagline { font-size: 0.95rem; color: #c9a878; line-height: 1.6; max-width: 520px; }
.hero-emoji { font-size: 2.8rem; margin-bottom: 0.6rem; display: block; }

.upload-label {
    font-size: 0.85rem; font-weight: 700; color: #3d2510;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 6px;
}

[data-testid="stFileUploader"] {
    background: #fdf6ec !important; border: 2.5px dashed #b8845a !important;
    border-radius: 14px !important; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #d97706 !important; }
[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploader"] section > div { background: transparent !important; color: #3d2510 !important; }
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] > div,
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #5a3820 !important; font-family: 'Nunito', sans-serif !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #d97706, #b86010) !important;
    color: #ffffff !important; border: none !important; border-radius: 9px !important;
    font-family: 'Nunito', sans-serif !important; font-weight: 700 !important;
    font-size: 0.85rem !important; padding: 8px 18px !important;
    box-shadow: 0 2px 8px #d9770640 !important;
}
[data-testid="stFileUploader"] button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

.book-card {
    background: #fdf0d8; border: 2px solid #c89050;
    border-left: 5px solid #d97706; border-radius: 12px;
    padding: 12px 18px; margin: 0.5rem 0;
    display: flex; align-items: center; gap: 12px;
    font-size: 0.88rem; color: #3d2510;
}
.book-card .book-title { font-weight: 700; color: #1c1408; font-size: 0.95rem; }
.book-pill {
    background: #d97706; border-radius: 20px; padding: 3px 12px;
    font-size: 0.76rem; color: #fff; font-weight: 700; white-space: nowrap;
}

.my-divider { border: none; border-top: 2px solid #c8a878; margin: 1.4rem 0; }

.msg-wrap { margin: 1rem 0; }
.msg-user {
    background: #2c1a08; border: 2px solid #5a3820;
    border-radius: 18px 18px 4px 18px; padding: 13px 18px;
    margin-left: 18%; font-size: 0.93rem; color: #f5e6c8; line-height: 1.65;
    box-shadow: 0 4px 16px #1c140830;
}
.msg-bot {
    background: #fdf6ec; border: 2px solid #c8a070;
    border-radius: 18px 18px 18px 4px; padding: 15px 20px;
    margin-right: 10%; font-size: 0.93rem; color: #1c1408; line-height: 1.75;
    box-shadow: 0 4px 16px #c8a07028; font-family: 'Lora', serif;
}
.msg-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 5px; font-family: 'Nunito', sans-serif;
}
.msg-label.you { color: #f59e0b; }
.msg-label.ai  { color: #7c5c3e; }

.source-card {
    background: #fdf6ec; border: 1.5px solid #c8a070;
    border-left: 4px solid #10b981; border-radius: 0 10px 10px 0;
    padding: 10px 14px; margin: 6px 0; font-size: 0.82rem; color: #3d2510; line-height: 1.6;
}
.source-card b { color: #059669; font-size: 0.72rem; font-family: 'Nunito', sans-serif; font-weight: 700; }

.highlight-card {
    background: #fffbeb; border: 2px solid #fbbf24;
    border-left: 5px solid #f59e0b; border-radius: 0 12px 12px 0;
    padding: 12px 16px; margin: 6px 0; font-size: 0.88rem; color: #1c1408; line-height: 1.7;
    font-family: 'Lora', serif;
}
.highlight-card b { color: #b45309; font-size: 0.72rem; font-family: 'Nunito', sans-serif; font-weight: 700; }

.empty-state { text-align: center; padding: 3.5rem 2rem; }
.empty-state .big-icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-state h3 { font-family: 'Lora', serif; font-size: 1.3rem; color: #3d2510; margin-bottom: 8px; font-weight: 600; }
.empty-state p { font-size: 0.88rem; color: #6b4530; line-height: 1.6; }

.suggest-label {
    font-size: 0.78rem; font-weight: 700; color: #6b4530;
    text-transform: uppercase; letter-spacing: 0.07em; margin: 1rem 0 0.5rem 0;
}

[data-testid="stSidebar"] { background: #e8d8c0 !important; border-right: 2px solid #c8a878; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-section { background: #fdf0d8; border: 2px solid #c8a070; border-radius: 12px; padding: 14px; margin-bottom: 14px; }
.sidebar-title { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; color: #7c5030; margin-bottom: 10px; }

[data-testid="stSelectbox"] > div > div {
    background: #fdf6ec !important; border: 2px solid #c8a070 !important;
    color: #1c1408 !important; border-radius: 10px !important; font-weight: 600 !important;
}
[data-testid="stSelectbox"] svg { color: #d97706 !important; }

.stButton > button {
    background: linear-gradient(135deg, #d97706, #b86010) !important;
    border: none !important; color: white !important; border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important; font-size: 0.85rem !important;
    font-family: 'Nunito', sans-serif !important; font-weight: 700 !important;
    cursor: pointer !important; width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 2px 10px #d9770640 !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

[data-testid="stChatInput"] textarea {
    background: #fdf6ec !important; border: 2px solid #c8a070 !important;
    border-radius: 14px !important; color: #1c1408 !important;
    font-family: 'Nunito', sans-serif !important; font-size: 0.93rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9a7050 !important; }
[data-testid="stChatInput"] textarea:focus { border-color: #d97706 !important; box-shadow: 0 0 0 3px #d9770625 !important; }

.stSuccess { background: #d1fae5 !important; border: 2px solid #10b981 !important; color: #064e3b !important; border-radius: 10px !important; font-weight: 600 !important; }
.stError { background: #fee2e2 !important; border: 2px solid #ef4444 !important; border-radius: 10px !important; color: #7f1d1d !important; }

[data-testid="stExpander"] { background: #fdf0d8 !important; border: 1.5px solid #c8a070 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #3d2510 !important; font-weight: 600 !important; }

[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    border: none !important; color: white !important; border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important; font-size: 0.85rem !important;
    font-family: 'Nunito', sans-serif !important; font-weight: 700 !important;
    width: 100% !important; box-shadow: 0 2px 10px #05966940 !important;
}

/* sidebar text fix - dark boxes need light color */
[data-testid="stSidebar"] *:not(.dark-box-text) {
    color: #1c1408 !important;
}
.sidebar-section * { color: #1c1408 !important; }

.dark-box-text {
    color: #c9a878 !important;
    font-family: 'Lora', serif !important;
    font-size: 0.80rem !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
}

[data-testid="stFileUploader"] * { color: #1c1408 !important; }
[data-testid="stFileUploader"] span { color: #1c1408 !important; }
.stSpinner div { color: #1c1408 !important; }

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)


EMBED_MODEL = "nomic-embed-text"

models = {
    "phi3":      "Phi-3  · Fast & light",
    "llama3.2":  "Llama 3.2  · Balanced",
    "mistral":   "Mistral  · Smart",
    "tinyllama": "TinyLlama  · Fastest",
}

depth_labels = {1: "Quick glance", 2: "Normal", 3: "Thorough", 4: "Deep dive", 5: "Full scan"}

# init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "_show_highlights" not in st.session_state:
    st.session_state._show_highlights = False


def load_pdf(pdf_file):
    tmp_path = f"_tmp_{pdf_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    pages = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

    for c in chunks:
        c.metadata["source_file"] = pdf_file.name

    vs = FAISS.from_documents(chunks, OllamaEmbeddings(model=EMBED_MODEL))
    os.remove(tmp_path)
    return vs, len(pages), len(chunks)


def get_merged_store():
    all_docs = []
    for d in st.session_state.documents.values():
        all_docs.extend(d["vectorstore"].docstore._dict.values())
    if not all_docs:
        return None
    return FAISS.from_documents(all_docs, OllamaEmbeddings(model=EMBED_MODEL))


def get_memory(n=4):
    hist = [m for m in st.session_state.chat_history if m["role"] in ("user", "assistant")]
    pairs = []
    for i in range(0, len(hist) - 1, 2):
        if i + 1 < len(hist):
            pairs.append(f"Student: {hist[i]['content']}\nAssistant: {hist[i+1]['content']}")
    return "\n\n".join(pairs[-n:])


def make_export():
    lines = [
        "# DocMind — Session Export",
        f"_{datetime.now().strftime('%d %b %Y, %I:%M %p')}_\n",
        f"**Docs:** {', '.join(st.session_state.documents.keys()) or '—'}\n",
        "---\n",
    ]
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            lines.append(f"### 🙋 You\n{msg['content']}\n")
        else:
            lines.append(f"### 📖 Answer\n{msg['content']}\n")
            if msg.get("sources"):
                lines.append("**Sources:**")
                for s in msg["sources"]:
                    doc_part = f" · {s['doc']}" if s.get("doc") else ""
                    lines.append(f"- Page {s['page']}{doc_part} — _{s['text'][:120]}…_")
                lines.append("")
    return "\n".join(lines)


# sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.2rem 0 1.8rem;'>
        <div style='font-size:2.4rem;'>📖</div>
        <div style='font-family:Lora,serif; font-weight:800; font-size:2.15rem; color:#2c2416; margin-top:6px;'>DocMind</div>
        <div style='font-size:0.73rem; color:#b89878; margin-top:4px;'>Your personal study companion</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🤖 Choose your assistant</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Assistant", list(models.keys()),
        format_func=lambda x: models[x],
        index=1, label_visibility="collapsed"
    )

    st.markdown("""
    <div style='margin-top:-6px; margin-bottom:12px; padding:9px 11px;
         background:#2c1a08; border-radius:8px; border:2px solid #5a3820;'>
        <p class='dark-box-text'>💡 Faster = shorter answers. Smarter = more thorough but slower.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🔎 How deep should it look?</div>', unsafe_allow_html=True)
    k_chunks = st.slider("Depth", 1, 5, 2, label_visibility="collapsed",
        help="Higher = reads more of your notes before answering.")
    st.caption(f"📏 {depth_labels.get(k_chunks, 'Normal')} — reads {k_chunks} section(s) of your notes")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🧠 Conversation memory</div>', unsafe_allow_html=True)
    use_memory = st.toggle("Remember previous Q&A", value=True,
        help="When ON, follow-up questions like 'explain that more' work naturally.")
    st.caption("Keeps context from your last 4 exchanges.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Your session</div>', unsafe_allow_html=True)

    docs = st.session_state.documents
    total_pages = sum(d["pages"] for d in docs.values())

    if docs:
        st.markdown(f"""
        <div class="sidebar-section">
            <div style='font-size:0.8rem; color:#6b4f35;'>
                📚 <b>{len(docs)} document(s) loaded</b><br>
                <span style='color:#b89878;'>{total_pages} pages · {st.session_state.total_questions} questions asked</span>
            </div>
        </div>""", unsafe_allow_html=True)

        for name, info in docs.items():
            st.markdown(f"""
            <div style='background:#fff8ed; border:1.5px solid #d4a96a; border-radius:8px;
                 padding:7px 12px; margin-bottom:6px; font-size:0.78rem; color:#3d2510;'>
                📄 <b>{name[:26]}{"…" if len(name) > 26 else ""}</b><br>
                <span style='color:#9a7050;'>{info["pages"]} pages · {info["chunks"]} sections</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size:0.82rem; color:#c9a882; padding:8px 0;'>
            No file loaded yet.<br>Upload a PDF to get started!
        </div>""", unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Export chat as Markdown",
            data=make_export(),
            file_name=f"DocMind_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )
        if st.button("🗑️ Start a fresh chat"):
            st.session_state.chat_history = []
            st.session_state.total_questions = 0
            st.rerun()

    st.markdown("""
    <div style='margin-top:auto; padding-top:2rem;'>
        <div style='background:#2c1a08; border:2px solid #5a3820; border-radius:10px; padding:12px 14px;'>
            <p class='dark-box-text'>
                ⚡ Runs 100% on your computer<br>
                🔒 Nothing leaves your device<br>
                🆓 Completely free to use
            </p>
        </div>
    </div>""", unsafe_allow_html=True)


# main area
st.markdown("""
<div class="hero-wrap">
    <span class="hero-emoji">📖</span>
    <h1>DocMind</h1>
    <p class="tagline">
        Upload any PDF — textbook, lecture notes, research paper — and have a real
        conversation with it. Ask questions, get summaries, clear up confusing parts.
        All on your own computer, for free.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-label">📂 Drop your study material here</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed",
    help="Upload one or more PDFs — add more anytime without restarting."
)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.documents]
    for pdf in new_files:
        with st.spinner(f"Reading '{pdf.name}'... just a moment ☕"):
            try:
                vs, page_count, chunk_count = load_pdf(pdf)
                st.session_state.documents[pdf.name] = {
                    "vectorstore": vs,
                    "pages": page_count,
                    "chunks": chunk_count,
                }
                st.success(f"✅ Done! Read all **{page_count} pages** of '{pdf.name}'. Ask me anything!")
            except Exception as e:
                st.error(f"⚠️ Couldn't load '{pdf.name}': {str(e)[:150]}")

docs = st.session_state.documents
store = get_merged_store()

if store:
    for fname, info in docs.items():
        st.markdown(f"""
        <div class="book-card">
            <span style='font-size:1.6rem;'>📘</span>
            <div>
                <div class="book-title">{fname}</div>
                <div style='color:#8a6d4e; font-size:0.8rem; margin-top:2px;'>
                    {info["pages"]} pages · Ready to answer your questions
                </div>
            </div>
            <div style='margin-left:auto; display:flex; gap:8px; flex-wrap:wrap;'>
                <span class="book-pill">{models[model_choice].split('·')[0].strip()}</span>
                <span class="book-pill">{depth_labels.get(k_chunks, 'Normal')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="my-divider">', unsafe_allow_html=True)

    st.markdown('<div class="suggest-label">⚡ Quick actions</div>', unsafe_allow_html=True)
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("✨ Short summary"):
            st.session_state._prefill = "__SHORT__"
            st.rerun()
    with qa2:
        if st.button("📋 Detail summary"):
            st.session_state._prefill = "__DETAILED__"
            st.rerun()
    with qa3:
        if st.button("🔎 Key highlights"):
            st.session_state._show_highlights = not st.session_state._show_highlights
            st.rerun()
    with qa4:
        if st.button("❓ Topics covered"):
            st.session_state._prefill = "What are all the main topics and concepts covered in this document?"
            st.rerun()

    if st.session_state._show_highlights:
        st.markdown('<div class="suggest-label">🔎 Important sections from your notes</div>', unsafe_allow_html=True)
        hits = store.similarity_search("important concept definition key idea main point", k=5)
        for i, h in enumerate(hits):
            pg = h.metadata.get("page", 0) + 1
            src = h.metadata.get("source_file", "")
            tag = f" · {src}" if src else ""
            st.markdown(f"""
            <div class="highlight-card">
                <b>📌 HIGHLIGHT {i+1} · PAGE {pg}{tag}</b><br>
                {h.page_content[:300]}{"…" if len(h.page_content) > 300 else ""}
            </div>""", unsafe_allow_html=True)
        if st.button("✖ Close highlights"):
            st.session_state._show_highlights = False
            st.rerun()
        st.markdown('<hr class="my-divider">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="big-icon">💬</div>
            <h3>What would you like to know?</h3>
            <p>Try asking about key topics, definitions, or ask me to explain something in simpler terms.</p>
        </div>
        """, unsafe_allow_html=True)
        
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-wrap">
                <div class="msg-label you">You asked</div>
                <div class="msg-user">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-wrap">
                <div class="msg-label ai">📖 Answer</div>
                <div class="msg-bot">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 See which parts of the document were used"):
                    for src in msg["sources"]:
                        pg = src["page"]
                        doc_tag = f"<b>{src['doc'].upper()} · </b>" if src.get("doc") else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <b>{doc_tag}PAGE {pg}</b><br>
                            {src["text"]}
                        </div>""", unsafe_allow_html=True)

    prefill = st.session_state.pop("_prefill", None)
    query = st.chat_input("Ask anything about your document... (e.g. 'Explain chapter 3 simply')")
    active_query = prefill or query

    if active_query:
        if active_query == "__SHORT__":
            display_q = "Give me a short summary of this document."
            prompt_q = "Give me a concise 3–5 sentence summary of the most important ideas in this document."
        elif active_query == "__DETAILED__":
            display_q = "Give me a detailed summary of this document."
            prompt_q = (
                "Give me a thorough, well-structured summary. Cover all major sections, "
                "key arguments, important definitions and conclusions. Use clear headings."
            )
        else:
            display_q = active_query
            prompt_q = active_query

        st.session_state.chat_history.append({"role": "user", "content": display_q})
        st.session_state.total_questions += 1

        with st.spinner("Reading your notes and thinking... 🤔"):
            try:
                results = store.similarity_search(prompt_q, k=k_chunks)

                ctx_parts = []
                for doc in results:
                    src = doc.metadata.get("source_file", "")
                    ctx_parts.append((f"[From: {src}]\n" if src else "") + doc.page_content)
                context = "\n\n---\n\n".join(ctx_parts)[:2200]

                memory_block = ""
                if use_memory:
                    mem = get_memory(n=4)
                    if mem:
                        memory_block = f"\n\nRecent conversation:\n{mem}\n"

                multi_doc_note = ""
                if len(docs) > 1:
                    multi_doc_note = f"You have {len(docs)} documents: {', '.join(docs.keys())}. Mention which document each piece of info comes from.\n"

                prompt = f"""You are a friendly, patient study assistant helping a student understand their notes.
{multi_doc_note}Answer ONLY using the document content below.
If the answer isn't there, say: "I couldn't find anything about that in your document. Try asking something else!"
Be clear, warm, and easy to understand — like explaining to a friend.
{memory_block}
Document content:
{context}

Student's question: {prompt_q}
Answer:"""

                llm = OllamaLLM(model=model_choice)
                answer = llm.invoke(prompt)

                sources = [
                    {
                        "page": doc.metadata.get("page", 0) + 1,
                        "doc": doc.metadata.get("source_file", ""),
                        "text": doc.page_content[:250] + "…",
                    }
                    for doc in results
                ]
                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer, "sources": sources
                })

            except Exception as e:
                err = str(e)
                if "connection" in err.lower() or "refused" in err.lower():
                    reply = "⚠️ The AI engine isn't running yet. Open a terminal and type `ollama serve`, then try again."
                else:
                    reply = f"⚠️ Something went wrong: {err[:200]}"
                st.session_state.chat_history.append({
                    "role": "assistant", "content": reply, "sources": []
                })

        st.rerun()

else:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">📂</div>
        <h3>No document loaded yet</h3>
        <p>
            Upload a PDF above to get started.<br>
            Textbooks, lecture slides, research papers — anything works!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin-top:1rem;'>
        <div style='background:#2c1a08; border:2px solid #5a3820; border-radius:14px; padding:18px 20px;
             flex:1; min-width:180px; max-width:220px; text-align:center;'>
            <div style='font-size:1.8rem; margin-bottom:10px;'>📤</div>
            <div style='font-weight:700; color:#f5e6c8; font-size:0.9rem;'>1. Upload your PDFs</div>
            <div style='color:#c9a878; font-size:0.78rem; margin-top:5px;'>Add multiple docs — mix textbooks, papers &amp; notes freely</div>
        </div>
        <div style='background:#2c1a08; border:2px solid #5a3820; border-radius:14px; padding:18px 20px;
             flex:1; min-width:180px; max-width:220px; text-align:center;'>
            <div style='font-size:1.8rem; margin-bottom:10px;'>💬</div>
            <div style='font-weight:700; color:#f5e6c8; font-size:0.9rem;'>2. Chat with memory</div>
            <div style='color:#c9a878; font-size:0.78rem; margin-top:5px;'>Ask follow-ups naturally — the assistant remembers your conversation</div>
        </div>
        <div style='background:#2c1a08; border:2px solid #5a3820; border-radius:14px; padding:18px 20px;
             flex:1; min-width:180px; max-width:220px; text-align:center;'>
            <div style='font-size:1.8rem; margin-bottom:10px;'>📄</div>
            <div style='font-weight:700; color:#f5e6c8; font-size:0.9rem;'>3. Export your notes</div>
            <div style='color:#c9a878; font-size:0.78rem; margin-top:5px;'>Save the full Q&amp;A session as a Markdown file to review anytime</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
