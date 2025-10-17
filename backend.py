from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
import chromadb
import pdfplumber
import requests
import logging
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import Counter

app = FastAPI()
logger = logging.getLogger("talk2dergipark")
logging.basicConfig(level=logging.INFO)
# Helper to normalize Gemini embedding responses
def _extract_embedding_values(embed_result):
    """Return embedding vector list from various Gemini response shapes."""
    logger.debug("embed_result type: %s", type(embed_result).__name__)
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


# CORS for chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# store collections in memory
collections_store = {}

# request models
class LoadPaperRequest(BaseModel):
    paper_url: str
    gemini_key: str

class AskQuestionRequest(BaseModel):
    paper_url: str
    question: str
    openai_key: str
    gemini_key: str

# helper functions
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

# backend functions
def get_pdf_url(paper_url: str) -> str:
    """scrape dergipark page to get pdf download link"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    try:
        response = requests.get(paper_url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as e:
        logger.exception("failed to fetch dergipark page: %s", e)
        raise HTTPException(status_code=502, detail=f"failed to fetch page: {e}")
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
    try:
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
    except Exception as e:
        logger.exception("failed to download pdf: %s", e)
        raise HTTPException(status_code=502, detail=f"failed to download pdf: {e}")
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

# API endpoints
@app.post("/load_paper")
async def load_paper(request: LoadPaperRequest):
    """
    load a dergipark paper
    returns: status, number of chunks
    """
    try:
        # get pdf url
        pdf_url = get_pdf_url(request.paper_url)
        if not pdf_url:
            raise HTTPException(status_code=404, detail="pdf link not found on page")

        # download and process
        filename = f"paper_{request.paper_url.split('/')[-1]}.pdf"
        pdf_path = download_pdf(pdf_url, filename)
        if not pdf_path:
            raise HTTPException(status_code=502, detail="failed to save pdf")

        text = extract_and_clean_pdf(pdf_path)
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=422, detail="no text extracted from pdf")

        chunks = chunk_with_langchain(text)
        if not chunks:
            raise HTTPException(status_code=422, detail="failed to create chunks from text")

        # embed and store
        collection = create_embeddings_and_store(chunks, request.paper_url, request.gemini_key)
        collections_store[request.paper_url] = collection

        return {
            "status": "success",
            "chunks": len(chunks),
            "message": "paper loaded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/load_paper failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: AskQuestionRequest):
    """
    ask a question about loaded paper
    returns: answer
    """
    try:
        if request.paper_url not in collections_store:
            raise HTTPException(status_code=404, detail="paper not loaded. load paper first")
        
        collection = collections_store[request.paper_url]
        
        # embed question
        genai.configure(api_key=request.gemini_key)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=request.question,
            task_type="retrieval_query"
        )
        query_vec = _extract_embedding_values(result)
        
        # retrieve relevant chunks
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=8
        )
        
        context = "\n\n---\n\n".join(results['documents'][0])
        
        # generate answer
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

        client = OpenAI(api_key=request.openai_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"paper content:\n\n{context}\n\n---\n\nquestion: {request.question}\n\nanswer:"}
            ],
            max_tokens=250,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """health check"""
    return {"message": "talk2dergipark api is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
