import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="AutoResearch AI", layout="wide")
st.title("🤖 AutoResearch AI Interface")

# --- Reset session button (optional) ---
if st.sidebar.button("🧹 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --- Initialize Session State ---
st.session_state.setdefault("uploaded_files", None)
st.session_state.setdefault("comparison_results", None)
st.session_state.setdefault("summary_limit", 2)
st.session_state.setdefault("summarized_results", [])
st.session_state.setdefault("filenames_input", "")
st.session_state.setdefault("rag_results", [])
st.session_state.setdefault("question_text", "")
st.session_state.setdefault("collection_names_input", "")
st.session_state.setdefault("rag_answer", "")
st.session_state.setdefault("selected_files", [])

# --- Sidebar for Route Selection ---
option = st.sidebar.radio("Select Operation", [
    "Upload & Compare PDFs",
    "Summarize Latest PDFs",
    "Ask Question (RAG)"
])

# === 1. Upload & Compare PDFs ===
if option == "Upload & Compare PDFs":
    st.header("📄 Upload 2–5 PDFs for Comparison & Novelty Detection")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.comparison_results = None  # reset old results

    if st.session_state.uploaded_files:
        if 2 <= len(st.session_state.uploaded_files) <= 5:
            if st.button("Upload and Analyze"):
                files = [("files", (f.name, f.read(), "application/pdf")) for f in st.session_state.uploaded_files]
                with st.spinner("Analyzing uploaded PDFs..."):
                    res = requests.post(f"{API_BASE}/upload-and-compare", files=files)
                if res.ok:
                    st.session_state.comparison_results = res.json()
                else:
                    st.error(res.text)

            # Show cached results
            if st.session_state.comparison_results:
                for paper in st.session_state.comparison_results:
                    st.subheader(paper["title"])
                    st.markdown(f"**Novel Insights:** {paper['novel_insights'][0]}")
                    st.markdown(f"**Similarities:** {paper['similarities'][0]}")
                    st.markdown(f"**Missing Gaps:** {paper['missing_gaps'][0]}")
        else:
            st.warning("Please upload between 2 to 5 PDFs.")
    else:
        st.info("Upload some PDFs to begin.")

# === 2. Summarize Latest PDFs ===
elif option == "Summarize Latest PDFs":
    st.header("📑 Summarize Latest Uploaded PDFs")

    limit = st.slider("How many recent PDFs?", min_value=1, max_value=5, value=st.session_state.get("summary_limit", 2))
    st.session_state.summary_limit = limit

    if st.button("🧠 Generate Summaries"):
        with st.spinner("Summarizing..."):
            res = requests.post(f"{API_BASE}/generate-summary/latest?limit={limit}")
        if res.ok:
            st.session_state.summarized_results = res.json()
        else:
            st.error(res.text)

    for item in st.session_state.get("summarized_results", []):
        st.subheader(item["title"])
        st.markdown(item["summary"])

# === 4. Ask Question (RAG) ===
elif option == "Ask Question (RAG)":
    st.header("❓ Ask a Question using Vector Search")

    question = st.text_input("Your Question:", value=st.session_state.get("question_text", ""))
    st.session_state.question_text = question

    # Fetch collection names from backend API once and cache
    @st.cache_data(ttl=600)
    def fetch_collections():
        try:
            res = requests.get(f"{API_BASE}/list-uploaded-collections")
            if res.ok:
                return res.json()
            else:
                st.error("❌ Failed to fetch collection names from database.")
                return []
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
            return []

    collections = fetch_collections()

    # Show multiselect for collections
    selected_collections = st.multiselect(
        "Select Collections to search in:",
        options=collections,
        default=st.session_state.get("selected_collections", [])
    )
    st.session_state.selected_collections = selected_collections

    if st.button("Ask"):
        if not question:
            st.warning("Please enter your question.")
        elif not selected_collections:
            st.warning("Please select at least one collection.")
        else:
            payload = {"question": question, "collection_names": selected_collections}
            with st.spinner("Searching for answer..."):
                res = requests.post(f"{API_BASE}/ask-question", json=payload)
            if res.ok:
                st.session_state.rag_answer = res.json().get("answer", "")
            else:
                st.error(res.text)

    if st.session_state.get("rag_answer"):
        st.success("✅ Answer:")
        st.markdown(st.session_state.rag_answer)

