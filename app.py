import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import chromadb
#from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import pdfplumber, re
from collections import Counter
# page config
st.set_page_config(
    page_title="Talk2Dergipark",
    page_icon="📚"
)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state['collection'] = None

# BACKEND FUNCTIONS
def get_pdf_url(paper_url: str) -> str:
    """get pdf download link from dergipark page"""
    response = requests.get(paper_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_link = soup.find('a', href=lambda x: x and 'article-file' in x)
    if pdf_link:
        return "https://dergipark.org.tr" + pdf_link['href']
    return None

def download_pdf(pdf_url: str) -> str:
    """download pdf"""
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open("paper.pdf", 'wb') as f:
            f.write(response.content)
        return "paper.pdf"
    return None

# def extract_text_from_pdf(pdf_path: str) -> str:
#     """extract text from pdf"""
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() + "\n\n"
#     return text
###############
def _normalize(line: str) -> str:
    # kill page-specific junk so variants cluster (e.g., "Vol 12 (2023) • 1140-1151" vs "… 1141-1151")
    line = re.sub(r'\d+', '<NUM>', line)          # numbers -> token
    line = re.sub(r'\s+', ' ', line).strip()      # collapse space
    line = line.upper()                           # case-insensitive match
    return line

def _top_bottom_lines(page_text: str, k_top=2, k_bottom=2):
    lines = [l for l in page_text.splitlines() if l.strip()]
    if not lines:
        return [], []
    head = lines[:min(k_top, len(lines))]
    tail = lines[-min(k_bottom, len(lines)):]
    return head, tail

def detect_repeating_banners(pdf_path: str, k_top=2, k_bottom=2, repeat_ratio=0.5):
    """
    Returns two sets of *normalized* lines that appear as headers/footers
    in >= repeat_ratio of pages (default: half the pages).
    """
    top_norms, bot_norms = [], []
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=1, y_tolerance=1) or ""
            pages_text.append(txt)

    for txt in pages_text:
        head, tail = _top_bottom_lines(txt, k_top, k_bottom)
        top_norms.extend(_normalize(l) for l in head if l.strip())
        bot_norms.extend(_normalize(l) for l in tail if l.strip())

    n_pages = max(1, len(pages_text))
    top_common = {l for l, c in Counter(top_norms).items() if c >= repeat_ratio * n_pages}
    bot_common = {l for l, c in Counter(bot_norms).items() if c >= repeat_ratio * n_pages}
    return top_common, bot_common, pages_text

def remove_headers_footers(pdf_path: str, k_top=2, k_bottom=2, repeat_ratio=0.5):
    """
    Extracts text and removes repeating header/footer lines across pages.
    Keeps everything else as-is (no aggressive cleanup that risks losing content).
    """
    top_common, bot_common, pages_text = detect_repeating_banners(
        pdf_path, k_top=k_top, k_bottom=k_bottom, repeat_ratio=repeat_ratio
    )

    cleaned_pages = []
    for txt in pages_text:
        lines = txt.splitlines()
        new_lines = []
        for i, line in enumerate(lines):
            norm = _normalize(line)
            # Decide if this line looks like a header/footer candidate by position
            is_top_zone = i < k_top
            is_bottom_zone = i >= len(lines) - k_bottom
            drop = (is_top_zone and norm in top_common) or (is_bottom_zone and norm in bot_common)
            if not drop:
                new_lines.append(line)
        cleaned_pages.append("\n".join(new_lines).strip())

    # light touch cleanup
    doc = "\n\n".join(p for p in cleaned_pages if p)
    doc = re.sub(r'\n{3,}', '\n\n', doc)
    doc = re.sub(r'[ \t]{2,}', ' ', doc)
    return doc
#################

def chunk_with_langchain(text: str):
    """chunk text with langchain"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def _extract_embedding_values(embed_result):
    """Return embedding vector list from various Gemini response shapes."""
    if isinstance(embed_result, dict):
        emb = embed_result.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list):
            return emb
        if "data" in embed_result and embed_result["data"]:
            sub = embed_result["data"][0].get("embedding")
            if isinstance(sub, dict) and "values" in sub:
                return sub["values"]
            if isinstance(sub, list):
                return sub
    raise ValueError("Unexpected embedding response format from Gemini")

def create_embeddings_and_store(chunks, paper_url, api_key):
    """embed chunks and store in chromadb"""
    genai.configure(api_key=api_key)
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk,
            task_type="retrieval_document"
        )
        embeddings.append(_extract_embedding_values(result))
    
    client = chromadb.Client()
    collection_name = "paper_" + paper_url.split('/')[-1]
    
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    
    return collection

def ask_question(collection, question, api_key):
    """ask question and get answer"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=question,
        task_type="retrieval_query"
    )
    
    query_vec = _extract_embedding_values(result)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=8
    )
    
    context = "\n\n---\n\n".join(results['documents'][0])
    
    system_prompt = """you are an academic paper assistant. you help users understand research papers.

your role:
- answer questions based on the paper content provided
- answer in the SAME language as the question (turkish → turkish, english → english)
- be specific: include exact names, numbers, methods, findings
- if info not in content, say: "this information is not available in the provided excerpt" or "bu bilgi verilen bölümde bulunmuyor"

guidelines:
1. only use provided content
2. be concise (2-4 sentences)
3. cite specific details
4. if unsure, say so"""

    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"paper content:\n\n{context}\n\n---\n\nquestion: {question}\n\nanswer:"}
        ],
        max_tokens=250,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# UI
st.title("📚 Talk2Dergipark")
st.write("chat with any dergipark paper in turkish or english")

# sidebar
with st.sidebar:
    st.header("API Keys")
    gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    st.markdown("---")
    st.write("enter your api keys to get started")

# main area
paper_url = st.text_input("Enter Dergipark Paper URL:", key="paper_url")

if st.button("Load Paper", key="load_paper"):
    if not paper_url or not gemini_key:
        st.error("please enter url and gemini api key")
    else:
        try:
            with st.spinner("processing paper..."):
                pdf_url = get_pdf_url(paper_url)
                if not pdf_url:
                    st.error("could not find a PDF link on the provided page")
                    st.stop()
                pdf_path = download_pdf(pdf_url)
                if not pdf_path:
                    st.error("failed to download PDF")
                    st.stop()
                #text = extract_text_from_pdf(pdf_path)
                text = remove_headers_footers(pdf_path)
                chunks = chunk_with_langchain(text)
                collection = create_embeddings_and_store(chunks, paper_url, gemini_key)
                
                st.session_state['collection'] = collection
                
            st.success(f"✅ paper loaded! {len(chunks)} chunks indexed")
        except Exception as e:
            st.error(f"error: {e}")

# question area
if 'collection' in st.session_state and st.session_state['collection'] is not None:
    question = st.text_input("Ask a question:", key="question")
    
    if st.button("Ask", key="ask") and question and openai_key:
        with st.spinner("thinking..."):
            try:
                answer = ask_question(st.session_state['collection'], question, openai_key)
                st.info(answer)
            except Exception as e:
                st.error(f"error: {e}")