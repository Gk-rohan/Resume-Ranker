############################################
# Resume Evaluation Assistant - Streamlit App
############################################

# ---------- Imports ----------
import os
import re
import base64
import pickle
import tempfile
import asyncio

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_pdf_viewer import pdf_viewer
from docx2pdf import convert

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

############################################
# ---------- Constants & Config ----------
############################################

SAMPLE_JOB_DESCRIPTION = """NVIDIA is looking for a senior technical program manager to lead new product introduction for hardware for NVIDIA Infrastructure Specialists (NVIS) team. We want you to collaborate with cross-functional teams, including professional services, solutions architects, development engineers, hardware and software engineering, data center operations, project managers, product managers, and go-to-market strategy teams. We want your primary focus is to ensure NVIS' readiness as NVIDIA introduces new hardware including deployment, provisioning and validation for early customers. You will be working with and have the support of the global NVIS team and in turn supporting the team as delivery transitions to production deployment.
                            What will you be doing:
                                •	Leading end-to-end execution of service programs related to new hardware product introduction and other related programs, ensuring adherence to project timelines, budgets, and quality standards. This includes applying your expertise to drive technical strategy, planning, and execution with the team, partners and customers.
                                •	Developing comprehensive program delivery plans to achieve successful project outcomes, including scoping, resource allocation, task sequencing, and risk management strategies.
                                •	Engaging and building internal and external customer relationships, understanding their needs and expectations, and effectively communicating program status, risks, and mitigation plans to ensure customer satisfaction. This includes engaging executives, engineering teams, and external partners and ensuring visibility and informed decision-making.
                                •	You will work with partners, decomposing requirements into technical execution plans, tracking progress towards goals, and reporting status to customers and technological leadership.
                                •	Establishing and maintaining project metrics and key performance indicators to track progress, evaluate program success, Identify areas for process improvement, and drive initiatives to improve service program.
                            What we need to see:
                                •	BS/MS Engineering or Computer Science (or equivalent experience)
                                •	12+ years of experience in project delivery management
                                •	Minimum 5 years of experience in providing field services and/or customer support for hardware & software products
                                •	In-depth knowledge of data center environments, servers, and network equipment
                                •	Strong interpersonal skills and the ability to work directly with customers
                                •	Supreme leadership skills across broad and diverse functional teams
                                •	Strong ability prioritize/multi-task easily with limited supervision
                                •	Experience leading global projects
                            NVIDIA is widely considered to be one of the technology world’s most desirable employers. We have some of the most forward-thinking and hardworking people in the world working for us. If you're creative and autonomous, we want to hear from you!"""
DOCS_DIR = os.path.abspath("./uploaded_docs")
VECTOR_STORE_PATH = "vectorstore.pkl"
RESUME_MAP_PATH = "resumemap.pkl"
VALID_CAND_PATH = "validcand.pkl"

############################################
# ---------- Page Setup ----------
############################################

api_key = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    layout="wide",
    page_title="Resume Evaluation Assistant", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

st.header('Resume Evaluation Assistant 🤖📝', divider='rainbow')

def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.markdown('''
Job listings currently receive hundreds of resumes. 
This system streamlines that process by leveraging the power of Google GEMINI
to evaluate resumes via a RAG pipeline.
Upload resumes, enter a job description, and get AI-based recommendations for top applicants. 
''')
st.warning("This is a proof of concept and should only be used to supplement traditional evaluation methods.", icon="⚠️")

############################################
# ---------- Helper Functions ----------
############################################

def extract_name(raw_output):
    """Extract candidate number, name, and description from model output."""
    if raw_output and raw_output[0].isdigit():
        number = str(re.match(r'^\d+', raw_output).group())
        candidate = raw_output.split(':')[0].replace('*', '').strip()
        while candidate and not candidate[0].isalpha():
            candidate = candidate[1:].lstrip()
        candidate = candidate.rstrip('.')
        description = raw_output.split(':', 1)[1] if ':' in raw_output else ''
    else:
        return "", "", ""
    return number, candidate, description

def save_uploaded_files(uploaded_files):
    """Save uploaded resumes to DOCS_DIR."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
        st.info(f"File {uploaded_file.name} uploaded successfully!")

############################################
# ---------- Sidebar: File Upload ----------
############################################

with st.sidebar:
    st.subheader("Upload Applicant Information")
    with st.form("upload-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload Resumes:", accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")
    if submitted and uploaded_files:
        save_uploaded_files(uploaded_files)

############################################
# ---------- LLM & Embedding Setup ----------
############################################

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=api_key
)

document_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)
query_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

############################################
# ---------- Vector Store Setup ----------
############################################

############################################
# ---------- Vector Store Setup ----------
############################################

FAISS_INDEX_PATH = "faiss_index"

use_existing_vector_store = st.sidebar.radio(
    "Use existing vector store if available", ["Yes", "No"], horizontal=True
)

name_extraction_prompt = PromptTemplate(
    input_variables=["resume_text", "file_name"],
    template="""Only output the full name of the candidate based on their resume...
    Filename: {file_name}
    Resume Content: {resume_text}
    Candidate name:"""
)
name_extraction_chain = (
    RunnablePassthrough.assign(
        resume_text=lambda x: x["resume_text"],
        file_name=lambda x: x["file_name"]
    )
    | name_extraction_prompt
    | llm
    | StrOutputParser()
)

raw_documents = DirectoryLoader(DOCS_DIR).load() if os.path.exists(DOCS_DIR) else []
vectorstore, resume_name_map, valid_candidates = None, {}, set()
first_doc = False

if use_existing_vector_store == "Yes" and os.path.exists(FAISS_INDEX_PATH):
    # Load FAISS index from disk
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, document_embedder, allow_dangerous_deserialization=True
    )
    with open(RESUME_MAP_PATH, "rb") as f:
        resume_name_map = pickle.load(f)
    with open(VALID_CAND_PATH, "rb") as f:
        valid_candidates = pickle.load(f)
    st.sidebar.info("Existing vector store loaded successfully.")

elif raw_documents:
    documents = []
    with st.spinner("Processing documents..."):
        for doc in raw_documents:
            try:
                filename = os.path.basename(doc.metadata.get('source', ''))
                candidate_name = name_extraction_chain.invoke({
                    "resume_text": doc.page_content,
                    "file_name": filename
                }).strip()
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata={"source": filename, "candidate_name": candidate_name}
                )
                resume_name_map[candidate_name.lower()] = filename
                valid_candidates.add(candidate_name.lower())
                documents.append(processed_doc)
            except Exception as e:
                st.warning(f"Error processing {filename}: {e}")

    with st.spinner("Adding documents to vector database..."):
        for doc in documents:
            if not first_doc:
                vectorstore = FAISS.from_documents([doc], document_embedder)
                first_doc = True
            else:
                vectorstore.add_documents([doc])

    with st.spinner("Saving FAISS index and metadata..."):
        vectorstore.save_local(FAISS_INDEX_PATH)
        with open(RESUME_MAP_PATH, "wb") as f:
            pickle.dump(resume_name_map, f)
        with open(VALID_CAND_PATH, "wb") as f:
            pickle.dump(valid_candidates, f)
        st.sidebar.success("Vector store saved.")


############################################
# ---------- Resume Evaluation ----------
############################################

valid_candidates_list = ', '.join(valid_candidates)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Based on the given job description..."),
    ("user", "Job Description: {input}\n\nCandidates: {context}\nValid names: " + valid_candidates_list)
])

job_description = st.text_area("Enter the job description:", value=SAMPLE_JOB_DESCRIPTION, height=350)
compressor = LLMChainExtractor.from_llm(llm)

if st.button("Evaluate Resumes") and vectorstore is not None and job_description:
    with st.spinner("Fetching resumes..."):
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 15})
        )
        docs = retriever.invoke(job_description)
        context = "".join(
            f"[CANDIDATE START] Candidate Name: {doc.metadata.get('candidate_name', 'Unknown')}\n"
            f"{doc.page_content}[CANDIDATE END]\n\n" for doc in docs
        )
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"input": job_description, "context": context})
    st.markdown("### Top Applicants:")

    found_any = False  # Track if any candidates matched
    candidates = [line.strip() for line in response.split("\n") if line.strip()]

    for idx, candidate in enumerate(candidates):
        number, case_correct_name, description = extract_name(candidate)
        lower_name = case_correct_name.lower()

        if number != "":
            found_any = True
            with stylable_container(
                key=f"container_with_border_{idx}",  # unique key
                css_styles="""
                    {
                        border: 0px solid #ccccd4;
                        border-radius: 0.75rem;
                        padding: calc(1em + 2px);
                        background-color: #f0f2f6
                    }
                """,
            ):
                st.markdown(f"##### {number}. {case_correct_name}")
                st.markdown(description)

                if lower_name in resume_name_map:
                    file_name = resume_name_map[lower_name]
                    file_path = os.path.join(DOCS_DIR, file_name)

                    with st.expander("View Resume"):
                        if os.path.exists(file_path):
                            if file_name.lower().endswith(('.doc', '.docx')):
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                    convert(file_path, tmp_pdf.name)
                                    pdf_path = tmp_pdf.name
                            else:
                                pdf_path = file_path
                            with open(pdf_path, "rb") as f:
                                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                            st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>', unsafe_allow_html=True)
                            if file_name.lower().endswith(('.doc', '.docx')):
                                os.unlink(pdf_path)
                        else:
                            st.warning("File not found.")
                else:
                    st.info("No resume available for this candidate.")
        else:
            # Still show the model's text (could be a reason/explanation)
            st.markdown(candidate)

    # If no top candidates found, give a reason
    if not found_any:
        st.info("No suitable applicants found based on the provided job description. "
                "The resumes may not match the required skills or experience.")

st.markdown("---")
st.markdown("<div class='footer'>Powered by <a href='https://gemini.google.com/'>GOOGLE GEMINI</a> | © 2025 <a href='https://www.linkedin.com/in/guptarohandec/'>Rohan Gupta</a></div>", unsafe_allow_html=True)
