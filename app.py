import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import chromadb
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import pdfplumber
from collections import Counter

# page config
st.set_page_config(
    page_title="Talk2Dergipark",
    page_icon="💬",
    layout="wide"
)

# Custom CSS to match Chrome extension style
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #a2083d 0%, #8a0734 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 12px rgba(162, 8, 61, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #a2083d 0%, #8a0734 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(162, 8, 61, 0.2);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(162, 8, 61, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #a2083d;
        box-shadow: 0 0 0 3px rgba(162, 8, 61, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: white;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #a2083d;
    }
    
    /* Info box styling */
    .stInfo {
        background-color: white;
        border: 1px solid #e9ecef;
        border-left: 4px solid #a2083d;
    }
    
    /* Success box styling */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-left: 4px solid #28a745;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #a2083d 0%, #8a0734 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        border-bottom-right-radius: 4px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(162, 8, 61, 0.2);
    }
    
    .assistant-message {
        background: white;
        color: #212529;
        padding: 1rem;
        border-radius: 12px;
        border-bottom-left-radius: 4px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state['collection'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# BACKEND FUNCTIONS (synced with backend.py)
def _normalize(line: str) -> str:
    """normalize line for comparison (numbers→<NUM>, uppercase, collapse spaces)"""
    line = re.sub(r'\d+', '<NUM>', line)
    line = re.sub(r'\s+', ' ', line).strip()
    line = line.upper()
    return line

def _top_bottom_lines(page_text: str, k_top=2, k_bottom=2):
    """get top k and bottom k lines from page"""
    lines = [l for l in page_text.splitlines() if l.strip()]
    if not lines:
        return [], []
    head = lines[:min(k_top, len(lines))]
    tail = lines[-min(k_bottom, len(lines)):]
    return head, tail

def detect_repeating_banners(pdf_path: str, k_top=2, k_bottom=2, repeat_ratio=0.5):
    """
    find lines that repeat as headers/footers across pages
    returns: (top_common, bot_common, pages_text)
    """
    top_norms, bot_norms = [], []
    pages_text = []

    # extract text from all pages
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=1, y_tolerance=1) or ""
            pages_text.append(txt)

    # collect normalized top/bottom lines
    for txt in pages_text:
        head, tail = _top_bottom_lines(txt, k_top, k_bottom)
        top_norms.extend(_normalize(l) for l in head if l.strip())
        bot_norms.extend(_normalize(l) for l in tail if l.strip())

    # find lines that appear in ≥ repeat_ratio of pages
    n_pages = max(1, len(pages_text))
    top_common = {l for l, c in Counter(top_norms).items() if c >= repeat_ratio * n_pages}
    bot_common = {l for l, c in Counter(bot_norms).items() if c >= repeat_ratio * n_pages}
    
    return top_common, bot_common, pages_text

def extract_and_clean_pdf(pdf_path: str, k_top=2, k_bottom=2, repeat_ratio=0.5) -> str:
    """
    extract text from pdf and remove repeating headers/footers
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
            is_top_zone = i < k_top
            is_bottom_zone = i >= len(lines) - k_bottom
            
            # drop if it's a repeating header/footer
            drop = (is_top_zone and norm in top_common) or (is_bottom_zone and norm in bot_common)
            
            if not drop:
                new_lines.append(line)
        
        cleaned_pages.append("\n".join(new_lines).strip())

    # join pages and cleanup
    doc = "\n\n".join(p for p in cleaned_pages if p)
    doc = re.sub(r'\n{3,}', '\n\n', doc)
    doc = re.sub(r'[ \t]{2,}', ' ', doc)
    return doc

def get_pdf_url(paper_url: str) -> str:
    """scrape dergipark page to get pdf download link"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    response = requests.get(paper_url, headers=headers, timeout=20)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_link = soup.find('a', href=lambda x: x and 'article-file' in x)
    if pdf_link:
        return "https://dergipark.org.tr" + pdf_link['href']
    return None

def download_pdf(pdf_url: str, filename: str = "paper.pdf") -> str:
    """download pdf from url"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    response = requests.get(pdf_url, headers=headers, timeout=60)
    response.raise_for_status()
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    return None

def chunk_with_langchain(text: str):
    """chunk text using recursive character splitter"""
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
        # Newer clients: {"embedding": {"values": [...]}}
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        # Sometimes: {"embedding": [...]} (list directly)
        if isinstance(emb, list):
            return emb
        # Occasionally batched: {"data":[{"embedding":{...} } ]}
        if "data" in embed_result and embed_result["data"]:
            sub = embed_result["data"][0].get("embedding")
            if isinstance(sub, dict) and "values" in sub:
                return sub["values"]
            if isinstance(sub, list):
                return sub
    # Some clients may return a raw list of floats already
    if isinstance(embed_result, list):
        # Either a flat vector or a batched list of vectors; pick first if nested
        if embed_result and isinstance(embed_result[0], list):
            return embed_result[0]
        return embed_result
    raise ValueError("Unexpected embedding response format from Gemini")

def create_embeddings_and_store(chunks, paper_url, api_key):
    """embed chunks and store in chromadb"""
    genai.configure(api_key=api_key)
    embeddings = []
    for chunk in chunks:
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

def ask_question(collection, question, gemini_key, openai_key):
    """ask question and get answer with improved system prompt from backend.py"""
    genai.configure(api_key=gemini_key)
    
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
    
    # IMPROVED SYSTEM PROMPT (synced with backend.py)
    system_prompt = """You are an expert academic research assistant analyzing scholarly papers.

CRITICAL LANGUAGE RULE:
- If the question is in TURKISH → respond in TURKISH
- If the question is in ENGLISH → respond in ENGLISH
- ALWAYS match the language of the question exactly

Your responsibilities:
1. Provide precise, evidence-based answers citing specific details from the paper
2. Reference exact methodologies, results, figures, and data when available
3. If information is not in the provided content, explicitly state:
   - English: "This information is not available in the provided paper excerpt."
   - Turkish: "Bu bilgi verilen makale bölümünde bulunmuyor."

Response guidelines:
- Be concise (2-4 sentences for simple questions, longer for complex analyses)
- Use specific names, numbers, and technical terms from the paper
- Maintain academic tone while being accessible
- When citing results, mention the section if identifiable (e.g., "In the Results section...")

Remember: ALWAYS respond in the SAME language as the question."""

    client = OpenAI(api_key=openai_key)
    
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

# UI - Header with custom styling
st.markdown("""
<div class="main-header">
    <h1>💬 talk2DergiPark</h1>
    <p>Chat with academic papers / Akademik makalelerle sohbet edin</p>
</div>
""", unsafe_allow_html=True)
# Header photo
st.image("url_demonstration.png", use_container_width=True)
# Sidebar
with st.sidebar:
    st.markdown("## 🔑 API Keys / API Anahtarları")
    gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    st.markdown("---")
    st.caption("To get API keys: / API anahtarları almak için: ")
    st.caption("• [Get Gemini API Key](https://aistudio.google.com/app/apikey)")
    st.caption("• [Get OpenAI API Key](https://platform.openai.com/api-keys)")

# Main area
paper_url = st.text_input("📚 Enter Dergipark Paper URL: / Dergipark makale URL'si giriniz:", key="paper_url", 
                         placeholder="https://dergipark.org.tr/tr/pub/...")

col1, col2 = st.columns([1, 4])
with col1:
    load_button = st.button("Load Paper / Makale yükle", key="load_paper", use_container_width=True)

if load_button:
    if not paper_url or not gemini_key:
        st.error("❌ Please enter URL and Gemini API key / Lütfen URL ve Gemini API anahtarı girin")
    else:
        try:
            with st.spinner("🔄 Processing paper... / Makale yükleniyor..."):
                pdf_url = get_pdf_url(paper_url)
                if not pdf_url:
                    st.error("❌ Could not find a PDF link on the provided page / PDF linki bulunamadı")
                    st.stop()
                
                pdf_path = download_pdf(pdf_url)
                if not pdf_path:
                    st.error("❌ Failed to download PDF / PDF indirelemedi ")
                    st.stop()
                
                text = extract_and_clean_pdf(pdf_path)
                if not text or len(text.strip()) == 0:
                    st.error("❌ No text extracted from PDF")
                    st.stop()
                
                chunks = chunk_with_langchain(text)
                if not chunks:
                    st.error("❌ Failed to create chunks from text")
                    st.stop()
                
                collection = create_embeddings_and_store(chunks, paper_url, gemini_key)
                st.session_state['collection'] = collection
                st.session_state['chat_history'] = []  # Reset chat history for new paper
                
            st.success(f"✅ Paper loaded successfully! {len(chunks)} chunks / Makale başarıyla yüklendi!  {len(chunks)} parça ")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Chat area
if st.session_state['collection'] is not None:
    st.markdown("---")
    st.markdown("### 💬 Ask Questions / Sorunuzu yazınız")
    
    # Display chat history
    if st.session_state['chat_history']:
        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # Question input with form to handle Enter key
    with st.form(key='question_form', clear_on_submit=True):
        question = st.text_area("Ask a question about the paper:", key="question",
                                placeholder="What is the main finding? / Ana bulgu nedir?",
                                height=100)
        submit_button = st.form_submit_button("Ask / Sor", use_container_width=True)
        
        if submit_button and question:
            if not openai_key:
                st.error("❌ Please enter OpenAI API key / Lütfen Gemini API anahtarı girin")
            else:
                with st.spinner("Thinking... / Düşünüyor... "):
                    try:
                        # Add user question to chat history
                        st.session_state['chat_history'].append({
                            'role': 'user',
                            'content': question
                        })
                        
                        # Get answer
                        answer = ask_question(st.session_state['collection'], question, gemini_key, openai_key)
                        
                        # Add assistant answer to chat history
                        st.session_state['chat_history'].append({
                            'role': 'assistant',
                            'content': answer
                        })
                        
                        # Rerun to update chat display
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
else:
    st.info(" Load a paper first to start asking questions / Soru sormak için makale yükleyiniz")