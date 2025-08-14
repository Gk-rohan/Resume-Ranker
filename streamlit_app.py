import os
import streamlit as st
import torch
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ──────────────────────────────────────────────────────────────────────────────
# 🔐 Configure API key
# Prefer: set in environment or Streamlit Secrets (st.secrets["GOOGLE_API_KEY"])
# Fallback to hardcoded variable ONLY for quick local testing.
# ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None) or "REPLACE_ME"

# Pick device safely
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Q&A Helper", page_icon="🧠", layout="wide")
st.title("Q&A Helper")

if API_KEY in (None, "", "REPLACE_ME"):
    st.error("Google API key is missing. Set `GOOGLE_API_KEY` in environment or in Streamlit Secrets.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Caches
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    print(f"[INIT] Loading embeddings on {DEVICE}...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

@st.cache_resource
def load_vector_db(vector_db_path: str):
    print("[INIT] Loading FAISS vector database...")
    embeddings = get_embeddings()
    vector_db = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("[READY] FAISS vector database loaded successfully.")
    return vector_db

@st.cache_resource
def get_llm():
    print("[INIT] Loading LLM (Gemini)...")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1,
        google_api_key=API_KEY,
    )

@st.cache_resource
def get_retriever():
    print("[INIT] Building retriever...")
    vector_db = load_vector_db("vector_db_faiss")
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever(search_kwargs={"k": 5})
    )
    print("[READY] Retriever ready.")
    return retriever

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
question = st.text_input("Enter your question:")
go = st.button("Ask", type="primary", use_container_width=True)

st.sidebar.header("About")
st.sidebar.info(
    "This is Q&A Helper. Enter your question and click 'Ask' to get an answer based on your FAISS-indexed documents."
)

# ──────────────────────────────────────────────────────────────────────────────
# Main action
# ──────────────────────────────────────────────────────────────────────────────
if go:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.status("Retrieving relevant documents…", expanded=False) as status:
        retriever = get_retriever()
        try:
            retrieved_docs = retriever.invoke(question)
            status.update(label="Documents retrieved ✓", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Retrieval failed", state="error", expanded=True)
            st.exception(e)
            st.stop()

    # Prepare context + doc info
    doc_info = [
        DocumentInfo(
            page=str(doc.metadata.get("page", "N/A")),
            link=doc.metadata.get("source", "N/A"),
            snippet=doc.page_content
        )
        for doc in retrieved_docs
    ]
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = prompt.format(context=context, question=question)

    # Stream answer
    st.subheader("Answer:")
    answer_placeholder = st.empty()
    streamed_text = ""

    llm = get_llm()
    try:
        # Stream tokens as they arrive
        for chunk in llm.stream(formatted_prompt):
            token = getattr(chunk, "content", None)
            if token is None:
                token = str(chunk)
            streamed_text += token
            # Render progressively
            answer_placeholder.markdown(streamed_text)
    except Exception as e:
        st.error("LLM streaming failed.")
        st.exception(e)
        st.stop()

    # Show retrieved docs
    st.subheader("Retrieved Documents:")
    if not doc_info:
        st.write("No documents were retrieved.")
    else:
        for doc in doc_info:
            with st.expander(f"Document (Page: {doc.page})"):
                st.write(f"Link: {doc.link}")
                st.write("Snippet:")
                st.write(doc.snippet)
