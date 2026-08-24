import os
from datetime import datetime

import requests
import streamlit as st


API_URL = os.getenv("SCHOLARRAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 240

st.set_page_config(
    page_title="ScholarRAG",
    page_icon="SR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #20211d;
        --muted: #6b6a5e;
        --bone: #f4f0e6;
        --paper: #fffdf7;
        --line: #ded8c9;
        --saffron: #d89532;
        --rust: #bd593d;
        --olive: #59634b;
    }

    .stApp { background: var(--bone); color: var(--ink); }
    [data-testid="stSidebar"] { background: #252820; border-right: 0; }
    [data-testid="stSidebar"] * { color: #f5f0e4; }
    [data-testid="stSidebar"] .stCaption { color: #bfc1ae; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { font-family: 'Space Grotesk', sans-serif; }
    h1 { letter-spacing: 0; font-size: 3.2rem; line-height: 1; margin-bottom: .35rem; }
    h2 { letter-spacing: 0; }
    .mono { font-family: 'DM Mono', monospace; }
    .eyebrow { color: var(--rust); font: 500 .72rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .hero { border-bottom: 1px solid var(--line); padding: 1.2rem 0 1.5rem; margin-bottom: 1.4rem; }
    .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 680px; }
    .stat { background: var(--paper); border: 1px solid var(--line); border-top: 3px solid var(--saffron); padding: 1rem 1.1rem; min-height: 100px; }
    .stat-value { color: var(--ink); font: 600 2rem 'Space Grotesk', sans-serif; }
    .stat-label { color: var(--muted); font: .7rem 'DM Mono', monospace; text-transform: uppercase; }
    .paper-row { border-bottom: 1px solid var(--line); padding: .85rem 0; }
    .paper-name { font-weight: 600; }
    .paper-meta { color: var(--muted); font: .74rem 'DM Mono', monospace; margin-top: .25rem; }
    .answer { background: var(--paper); border-left: 4px solid var(--olive); padding: 1.1rem 1.25rem; margin: .8rem 0 1rem; }
    .source { background: #ebe5d7; border: 1px solid var(--line); padding: .7rem .85rem; margin: .45rem 0; }
    .source-title { color: var(--rust); font-weight: 600; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 2px; border: 1px solid var(--ink); background: var(--ink); color: #fffdf7; font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
    div.stButton > button:hover { border-color: var(--rust); background: var(--rust); color: white; }
    .stTextInput input, .stTextArea textarea { background: var(--paper); border: 1px solid var(--line); border-radius: 2px; color: var(--ink); }
    .stFileUploader { background: var(--paper); border: 1px dashed var(--rust); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method, path, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        if response.ok:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error {response.status_code}: {detail}")
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach ScholarRAG API at {API_URL}: {exc}") from exc


def load_documents():
    return api_request("GET", "/documents").get("documents", [])


def format_date(value):
    if not value:
        return "Date unavailable"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value)[:10]


if "documents" not in st.session_state:
    try:
        st.session_state.documents = load_documents()
    except RuntimeError as exc:
        st.session_state.documents = []
        st.session_state.startup_error = str(exc)

with st.sidebar:
    st.markdown("<div class='eyebrow'>Research workspace</div>", unsafe_allow_html=True)
    st.markdown("# ScholarRAG")
    st.caption("A focused reading room for asking better questions of your papers.")
    st.divider()
    st.markdown("**API endpoint**")
    st.code(API_URL, language=None)
    if st.button("Refresh library", use_container_width=True):
        try:
            st.session_state.documents = load_documents()
            st.toast("Library refreshed")
        except RuntimeError as exc:
            st.error(str(exc))

st.markdown("<div class='hero'><div class='eyebrow'>Literature intelligence</div><h1>Ask your papers<br>something difficult.</h1><div class='hero-copy'>Upload research, build a searchable library, and get answers grounded in the documents you choose.</div></div>", unsafe_allow_html=True)

if st.session_state.get("startup_error"):
    st.warning(st.session_state.startup_error)

# Upload stays close to the library because indexing changes what can be asked.
left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown("## Library")
    documents = st.session_state.documents
    total_chunks = sum(doc.get("chunks", 0) or 0 for doc in documents)
    stat_one, stat_two = st.columns(2)
    with stat_one:
        st.markdown(f"<div class='stat'><div class='stat-value'>{len(documents)}</div><div class='stat-label'>Indexed papers</div></div>", unsafe_allow_html=True)
    with stat_two:
        st.markdown(f"<div class='stat'><div class='stat-value'>{total_chunks:,}</div><div class='stat-label'>Searchable chunks</div></div>", unsafe_allow_html=True)
    st.write("")
    if documents:
        for document in documents:
            status = document.get("status", "unknown")
            st.markdown(
                f"<div class='paper-row'><div class='paper-name'>{document.get('filename', 'Untitled paper')}</div><div class='paper-meta'>{document.get('chunks', 0)} chunks  /  {format_date(document.get('upload_date'))}  /  {status}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No papers indexed yet. Add a PDF to open the library.")

with right:
    st.markdown("## Add a paper")
    st.caption("PDF files are processed and indexed by the FastAPI backend.")
    uploaded_file = st.file_uploader("Choose a research paper", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.markdown(f"**{uploaded_file.name}**  ·  {uploaded_file.size / 1024:.0f} KB")
        if st.button("Upload and index", type="primary", use_container_width=True):
            with st.spinner("Extracting, chunking, and indexing..."):
                try:
                    result = api_request(
                        "POST",
                        "/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    )
                    st.success(result.get("message", "Paper indexed successfully."))
                    st.session_state.documents = load_documents()
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

st.divider()
st.markdown("## Ask the library")
if not st.session_state.documents:
    st.caption("Index at least one paper before asking a question.")
else:
    document_options = {doc["filename"]: doc["document_id"] for doc in st.session_state.documents}
    selected_names = st.multiselect(
        "Use every paper or narrow the question to selected papers",
        options=list(document_options),
        placeholder="All indexed papers",
    )
    question = st.text_area(
        "Question",
        placeholder="Compare the methods, limitations, and main contributions across these papers...",
        height=110,
        label_visibility="collapsed",
    )
    ask_col, hint_col = st.columns([1, 2])
    with ask_col:
        ask_clicked = st.button("Ask ScholarRAG", type="primary", use_container_width=True)
    with hint_col:
        st.caption("Answers are generated from retrieved passages and include their paper, section, and page.")

    if ask_clicked:
        if not question.strip():
            st.warning("Write a question first.")
        else:
            with st.spinner("Searching the library and composing an answer..."):
                try:
                    result = api_request(
                        "POST",
                        "/chat",
                        json={
                            "question": question.strip(),
                            "document_ids": [document_options[name] for name in selected_names] or None,
                        },
                    )
                    st.markdown("### Answer")
                    st.markdown(f"<div class='answer'>{result.get('answer', 'No answer returned.')}</div>", unsafe_allow_html=True)
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander(f"Sources ({len(sources)})", expanded=True):
                            for source in sources:
                                page = source.get("page")
                                page_text = f"Page {page + 1}" if isinstance(page, int) else "Page unavailable"
                                st.markdown(
                                    f"<div class='source'><div class='source-title'>{source.get('paper', 'Unknown paper')}</div><div class='paper-meta'>{source.get('section', 'Unknown section')}  /  {page_text}</div></div>",
                                    unsafe_allow_html=True,
                                )
                except RuntimeError as exc:
                    st.error(str(exc))
