# talk2DergiPark 🎓💬

> A RAG-powered Chrome extension for conversational interaction with Turkish academic papers from DergiPark


<a href="https://www.youtube.com/watch?v=r3IoqzohBo0">
  <img width="800" alt="Demo Video" src="https://github.com/user-attachments/assets/8ca27159-bdbb-4c97-a79e-3f0e1838a3ba" />
</a>

**[🎥 Watch full demo on YouTube →](https://www.youtube.com/watch?v=r3IoqzohBo0)**

### **Chrome Extension:** Available in this repository 


### **Developer:** [@esineyildiz](https://github.com/esineyildiz) 

[Coming Soon - Streamlit Deployment] 

---

## 📌 Project Overview

**talk2DergiPark** enables researchers to have natural language conversations with academic papers from [DergiPark](https://dergipark.org.tr), Turkey's largest academic publishing platform. Built for the **Akbank GenAI Bootcamp** and designed to showcase practical applications of LLMs in research workflows, this project demonstrates end-to-end implementation of a Retrieval-Augmented Generation (RAG) system with a novel browser-based interface.

### Inspiration

Inspired by [talk2arxiv](https://github.com/evanhu1/talk2arxiv), I wanted to create a similar tool tailored for Turkish academia, with enhanced PDF processing capabilities and a seamless browser integration via Chrome's Side Panel API.

### Why This Matters

Academic literature review is time-intensive. Researchers often need to quickly extract specific information from papers—methodology details, results, citations—without reading entire documents. This tool accelerates that process by:
- **Automated document processing** with intelligent header/footer removal
- **Semantic search** over paper content using state-of-the-art embeddings
- **Context-aware question answering** with multilingual support (Turkish/English)
- **Persistent side panel** for uninterrupted research workflow

---

## 🎯 Key Features

### 1. **Intelligent PDF Processing**
- **Automated header/footer detection and removal** using frequency analysis across pages
- **Smart text extraction** with proper handling of academic paper formatting
- **Recursive chunking** with semantic overlap for context preservation

### 2. **Advanced RAG Pipeline**
- **Embedding Model:** Google's `text-embedding-004` for document and query embeddings
- **Vector Database:** ChromaDB for efficient similarity search
- **Retrieval:** Top-8 semantic chunks with relevance ranking
- **Generation:** GPT-4o-mini with custom academic assistant prompt

### 3. **Bilingual Support**
- Automatic language detection and response in the query language
- Optimized prompts for both Turkish and English academic content

### 4. **Modern Chrome Extension UI**
- **Chrome Side Panel API** integration for persistent, distraction-free interface
- Custom DergiPark-themed design with gradient styling
- Real-time chat interface with loading states and error handling

---

## 🏗️ Technical Architecture
```
┌───────────────────┐
│  Chrome Extension │  (Manifest V3, Side Panel API)
│   popup.html/js   │
└────────┬──────────┘
         │ HTTP/REST
         ▼
┌──────────────────┐
│  FastAPI Backend │  (Python 3.10+)
│   backend.py     │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌──────────┐
│Gemini│  │ OpenAI   │
│ API  │  │ GPT-4o   │
└──────┘  └──────────┘
    │         │
    └────┬────┘
         ▼
    ┌─────────┐
    │ChromaDB │  (In-memory vector store)
    └─────────┘
```

### Core Components

**1. Document Processing Pipeline**
```python
# Smart header/footer detection using statistical analysis to improve chunking and retrieval 
def detect_repeating_banners(pdf_path, k_top=2, k_bottom=2, repeat_ratio=0.5):
    # Analyzes normalized text lines across all pages
    # Removes lines appearing in ≥50% of pages (configurable threshold)
```

**2. Embedding & Storage**
```python
# Google Gemini embeddings with task-specific optimization
genai.embed_content(
    model="models/text-embedding-004",
    content=chunk,
    task_type="retrieval_document"  # or "retrieval_query"
)
```

**3. Retrieval Strategy**
- **Chunk size:** 1000 characters
- **Overlap:** 200 characters (20% for context continuity)
- **Retrieval depth:** 8 most relevant chunks
- **Separators:** Semantic boundaries (`\n\n`, `\n`, `.`, ` `)

**4. Prompt Engineering**
```python
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
```

Key design decisions:
- **Language-agnostic**: Responds in query language automatically
- **Grounded responses**: Explicitly acknowledges when information isn't available
- **Conciseness**: 2-4 sentence limit to prevent hallucination and maintain focus

---

## 🔬 RAG Implementation Deep Dive

### Why This Approach?

1. **Gemini for Embeddings:** 
   - `text-embedding-004` offers 768-dimensional embeddings optimized for semantic search
   - Free tier available and simple API integration
   - Strong multilingual Performance: Specifically optimized for cross-lingual semantic similarity
   - Task-specific modes (`retrieval_document` vs `retrieval_query`) improve relevance

2. **ChromaDB for Storage:**
   - Simple in-memory operation for fast prototyping
   - Built-in cosine similarity search
   - Easy migration to persistent storage for production

3. **GPT-4o-mini for Generation:**
   - Balance between quality and cost
   - Strong multilingual capabilities (crucial for Turkish content)
   - Low latency for interactive chat experience

4. **Custom Text Cleaning:**
   - Academic PDFs have unique challenges (headers, footers, page numbers)
   - Frequency-based detection removes repetitive elements without hardcoding
   - Preserves paper structure while eliminating noise
     
---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get here](https://ai.google.dev/))
- OpenAI API key ([Get here](https://platform.openai.com/))
- Chrome browser (for extension)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/esineyildiz/talk2dergipark.git
cd talk2dergipark
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the backend**
```bash
uvicorn backend:app --reload
```

Backend will be available at `http://127.0.0.1:8000`

### Chrome Extension Setup

1. **Open Chrome Extensions**
   - Navigate to `chrome://extensions/`
   - Enable "Developer mode" (top-right toggle)

2. **Load extension**
   - Click "Load unpacked"
   - Select the `chrome-extension` folder from this repo

3. **Pin the extension**
   - Click the puzzle icon in Chrome toolbar
   - Pin "talk2DergiPark" for easy access

### Usage

1. Navigate to any DergiPark paper (e.g., `https://dergipark.org.tr/en/pub/...`)
2. Click the extension icon to open the side panel
3. Enter your API keys (stored locally, one-time setup)
4. Click "Load Paper" to process the current paper
5. Start asking questions in Turkish or English!

---

## 📊 Project Structure
```
talk2dergipark/
├── chrome-extension/
│   ├── manifest.json          # Chrome extension configuration
│   ├── popup.html             # Side panel UI
│   ├── popup.js               # Frontend logic
│   ├── background.js          # Service worker for side panel
│   └── content.js             # Content script (future: in-page annotations)
├── backend.py                 # FastAPI server with RAG pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore
```

---

## 🔧 Technologies Used

### Backend
- **FastAPI** - Modern async web framework
- **LangChain** - Text splitting utilities
- **ChromaDB** - Vector database
- **pdfplumber** - PDF text extraction
- **BeautifulSoup4** - Web scraping for PDF links
- **Google Gemini API** - Embeddings
- **OpenAI API** - Text generation

### Frontend
- **Chrome Extension Manifest V3** - Latest extension standard
- **Chrome Side Panel API** - Persistent UI
- **Vanilla JavaScript** - No framework overhead
- **Chrome Storage API** - Secure local key storage

### Development
- **Python 3.10+** - Type hints, async/await
- **uvicorn** - ASGI server
- **CORS middleware** - Cross-origin requests

---

## 🚀 Future Enhancements

### Short-term
- [ ] **Chat memory** - Keep previous messages and context
- [ ] **Persistent storage** - Save processed papers across sessions
- [ ] **Multi-paper chat** - Compare and contrast multiple papers

### Medium-term
- [ ] **Fine-tuned embeddings** - Domain-specific Turkish academic embeddings
- [ ] **Visual annotations** - Highlight relevant sections in original PDF

### Long-term
- [ ] **Graph-based retrieval** - Knowledge graph from citation networks
- [ ] **Integration with reference managers** - Zotero, Mendeley plugins

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Akbank GenAI Bootcamp** - Project motivation and learning opportunity
- **[talk2arxiv](https://github.com/evanhu1/talk2arxiv)** - Original inspiration
- **DergiPark** - Turkey's open access publishing platform

---

## 📧 Contact

**Esin Ezgi Yildiz** - [@esineyildiz](https://github.com/esineyildiz) 

📧 Email: esinezgiyildiz@gmail.com 

🔗 [LinkedIn](https://www.linkedin.com/in/esin-ezgi-yildiz/)  

