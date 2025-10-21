# talk2DergiPark: Bilingual RAG System for Turkish Academic Papers

**Author:** Esin Ezgi Yildiz  
**Project Type:** Chrome Extension + RAG Pipeline  
**Status:** Proof-of-Concept with Qualitative Evaluation

---

## 1. Introduction

**talk2DergiPark** is a Chrome extension that enables natural language conversations with academic papers from DergiPark, Turkey's largest open-access academic publishing platform. The system implements a Retrieval-Augmented Generation (RAG) pipeline optimized for bilingual (English-Turkish) question answering over scholarly documents.

This document provides a technical overview of the system architecture and presents results from a qualitative evaluation conducted to assess performance across languages and question types.

---

## 2. System Architecture

### 2.1 Overview

The system consists of three primary components: (1) a Chrome extension frontend using the Side Panel API, (2) a FastAPI backend implementing the RAG pipeline, and (3) integration with Google Gemini and OpenAI APIs for embeddings and generation respectively.

```
┌─────────────────────┐
│ Chrome Extension    │
│ (Side Panel UI)     │
└──────────┬──────────┘
           │ REST API
           ↓
┌──────────────────────┐
│ FastAPI Backend      │
│ (RAG Orchestration)  │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ↓             ↓
┌─────────┐  ┌──────────┐
│ Gemini  │  │ OpenAI   │
│ API     │  │ GPT-4o   │
└────┬────┘  └─────┬────┘
     │             │
     └──────┬──────┘
            ↓
     ┌──────────┐
     │ChromaDB  │
     │(Vectors) │
     └──────────┘
```

### 2.2 Document Processing Pipeline

The document processing pipeline addresses unique challenges in academic PDF extraction:

1. **PDF Acquisition**: Web scraping of DergiPark pages to extract PDF URLs
2. **Text Extraction**: pdfplumber-based extraction with layout preservation
3. **Banner Removal**: Statistical frequency analysis identifies and removes repeating headers/footers appearing in ≥50% of pages
4. **Semantic Chunking**: LangChain RecursiveCharacterTextSplitter with 1000-character chunks and 200-character overlap

The banner detection algorithm is particularly important for academic papers, which often contain institutional headers, page numbers, and journal footers that add noise to the embedding space.

### 2.3 Retrieval System

The retrieval system uses Google's `text-embedding-004` model to generate 768-dimensional embeddings for document chunks and queries. The model offers task-specific optimization modes: `retrieval_document` for indexing paper content and `retrieval_query` for processing user questions. This embedding model was selected for its strong multilingual capabilities, particularly for Turkish, and its free tier availability.

Embeddings are stored in ChromaDB, an in-memory vector database that performs cosine similarity-based retrieval. For each user query, the system retrieves the top-8 most relevant chunks. The retrieval strategy relies on dense retrieval without reranking, trusting the embedding quality to surface relevant content.

### 2.4 Generation System

The generation component uses OpenAI's GPT-4o-mini model, selected for its balance of output quality, response latency, and cost-effectiveness. The model's strong multilingual performance is particularly important for handling Turkish academic content alongside English.

The prompting strategy employs a language-aware system prompt with explicit instructions. The prompt directs the model to match the response language to the query language, ensuring Turkish questions receive Turkish answers and English questions receive English responses. Additionally, the prompt emphasizes providing evidence-based answers with specific citations from the paper, explicitly stating when requested information is unavailable, and maintaining conciseness with 2-4 sentence responses for straightforward queries. This conservative approach prioritizes accuracy over verbosity, reducing hallucination risk.

### 2.5 Interface Design

The Chrome Side Panel API provides persistent, non-intrusive access to the chat interface while users browse DergiPark. This design choice allows researchers to keep the assistant available without switching tabs or windows. User API keys are stored locally using the Chrome Storage API, ensuring credentials never leave the user's machine. The interface implements real-time loading states during document processing and query handling, along with comprehensive error messaging for network issues or API failures.

---

## 3. Evaluation Methodology

### 3.1 Evaluation Framework

A qualitative evaluation was conducted following established practices in NLP system assessment. While automated metrics like BLEU and ROUGE are common for translation and summarization tasks, generation quality for question-answering requires human judgment to assess semantic correctness, factual accuracy, and contextual appropriateness.

Each response was evaluated across five dimensions using a 5-point Likert scale. Fluency measures grammatical correctness and natural language quality. Relevance assesses the degree to which the answer addresses the question. Accuracy evaluates factual correctness relative to the source document, serving as a hallucination detector. Completeness examines the sufficiency of detail and coverage provided. Finally, language appropriateness considers correct language usage and domain-specific terminology.

Beyond dimensional scoring, responses were analyzed using an error taxonomy. This systematic classification identified specific failure modes including hallucination (fabricated information), omission (missing relevant details), mistranslation, incoherence, wrong scope (answering a different question), citation failure, repetition, overgeneralization (claims broader than source supports), context loss, and format errors.

### 3.2 Test Design

The evaluation used 2 papers as test cases: one English-source document and one Turkish-source document. Each paper was evaluated with 15 questions, totaling 30 questions across the study. The question distribution included 8 English-language questions and 7 Turkish-language questions per paper.

Questions were designed to cover eight categories representing typical research paper queries. These included identifying research objectives, describing methodology, extracting results, determining sample size, discussing limitations, analyzing contributions to literature, exploring practical implications, and handling out-of-scope questions (information not present in the paper).

It is important to note that this represents a proof-of-concept evaluation. Production-grade assessment would require 20-50+ papers across diverse academic domains to establish statistical significance and generalizability. The current scale serves to demonstrate the evaluation methodology and identify preliminary performance patterns.

### 3.3 LLM-as-Judge Methodology

Recent work in NLP evaluation has shown that large language models can serve as consistent, scalable evaluators for generation tasks. This study employs a hybrid approach combining automated and human evaluation. ChatGPT-4o served as the primary evaluator, rating all responses using the structured rubric described above. Following automated evaluation, manual review of all ratings was conducted to verify accuracy and identify edge cases that might confuse the LLM judge.

This hybrid methodology provides several advantages. It combines the scalability and consistency of automated evaluation with the reliability and nuanced judgment of human oversight. The approach mitigates known issues with LLM judges, such as positional bias and verbosity bias, while enabling rapid iteration during system development.

---

## 4. Results

### 4.1 Overall Performance

#### Table 1: Performance Comparison by Source Language

| Dimension | Paper 1 (EN) | Paper 2 (TR) | Δ |
|-----------|--------------|--------------|---|
| Fluency | 5.00 | 5.00 | 0.00 |
| Relevance | 5.00 | 4.80 | -0.20 |
| Accuracy | 4.87 | 4.00 | **-0.87** |
| Completeness | 4.73 | 3.80 | **-0.93** |
| Language Appropriateness | 5.00 | 4.47 | -0.53 |
| **Overall Average** | **4.92** | **4.41** | **-0.51** |

### 4.2 Detailed Findings

The English source document (Paper 1) achieved an overall score of 4.92/5.0, representing 98.4% quality. The system demonstrated perfect fluency and relevance, with minimal hallucinations detected. Error analysis revealed only 1 omission and 1 overgeneralization across all responses. When examining performance by question language, English questions scored 4.97 while Turkish questions scored 4.86, showing balanced bilingual capability.

The Turkish source document (Paper 2) achieved an overall score of 4.41/5.0, representing 88.2% quality. While the system maintained perfect fluency, other dimensions showed degradation. Notably, 5 omissions and 2 format errors were identified. Interestingly, the language performance pattern reversed: Turkish-language questions scored 4.63 while English-language questions scored 4.23. This suggests the system handles native-language queries more effectively when processing Turkish source documents.

### 4.3 Category Analysis

Performance varied significantly across question categories. High-performing categories (scoring ≥4.7/5.0) included research objective identification, methodology description, results extraction, and out-of-scope question handling, with the latter achieving perfect scores. Lower-performing categories (scoring <4.0/5.0) included sample size extraction (ranging from 3.80 to 4.70 across papers) and practical implications analysis (3.20 on Paper 2). Complex analytical questions requiring multi-hop reasoning consistently scored lower than straightforward factual queries.

### 4.4 Error Analysis

The error type distribution revealed omission as the most common failure mode, occurring 6 times total and particularly frequently on the Turkish paper. Format errors occurred twice, both on Paper 2, indicating issues with response structure. One instance of overgeneralization was detected, where the system made claims broader than the source material supported. Critically, no instances of fabricated information (hallucination) were detected, suggesting the grounding mechanism—which explicitly states when information is unavailable—functions effectively.

---

## 5. Discussion

### 5.1 Performance Gap Analysis

The 10.4 percentage point gap between English source documents (4.92) and Turkish source documents (4.41) warrants investigation. Several factors may contribute to this disparity. First, `text-embedding-004` may be optimized primarily for English despite multilingual claims, resulting in lower-quality embeddings for Turkish text. Second, Turkish academic papers may follow different formatting conventions, potentially affecting text extraction quality during the PDF processing stage. Third, GPT-4o-mini's training corpus likely contains more English academic content, giving it stronger domain knowledge for English papers. Finally, fixed-length chunking strategies may not align well with Turkish linguistic units, potentially breaking semantic coherence in ways that affect retrieval quality.

### 5.2 Completeness vs Accuracy Trade-off

The system demonstrates a clear prioritization of accuracy over completeness. When information is not available in the retrieved context, the system explicitly states unavailability rather than attempting to generate plausible but potentially incorrect responses. This conservative approach successfully minimizes hallucination risk, as evidenced by zero critical hallucinations detected. However, this design choice results in lower completeness scores, particularly on Paper 2 where completeness averaged 3.80/5.0. The trade-off reflects a deliberate decision to maintain reliability at the cost of comprehensive coverage.

### 5.3 Out-of-Scope Handling

The perfect scores (5.0/5.0) achieved on out-of-scope questions demonstrate robust refusal capability. When asked questions whose answers are not present in the paper—such as personal details about authors or information beyond the document scope—the system correctly identifies the limitation and declines to answer. This behavior represents a critical feature for academic applications, where providing ungrounded speculation could mislead researchers or misrepresent scholarly work.

### 5.4 Evaluation Validity

The evaluation approach demonstrates several strengths. The structured rubric ensures consistency across responses, while the LLM-as-judge methodology provides scalability for testing multiple papers. Human validation of all ratings ensures reliability and catches edge cases that might confuse automated evaluators. The error taxonomy enables systematic identification of failure patterns, facilitating targeted improvements.

However, the evaluation also has notable limitations. The small sample size of 2 papers limits generalizability of findings. Using a single human validator means inter-annotator agreement cannot be calculated, leaving questions about evaluation reliability unanswered. The absence of baseline comparisons—against other RAG systems or human expert performance—makes it difficult to assess whether observed performance is competitive. Finally, the domain limitation to academic papers means system behavior on other document types remains unknown.

---

## 6. Recommendations

### 6.1 System Improvements

Several immediate improvements could be implemented within 0-2 weeks. Adjusting the generation prompt to encourage more complete answers while maintaining grounding requirements could address the completeness gap without sacrificing accuracy. Implementing a chunk citation mechanism would allow users to see which sections of the paper informed each answer, increasing transparency and trust. Fine-tuning the numerical entity extraction pipeline specifically for sample sizes and statistical values would improve performance on quantitative questions.

Medium-term improvements spanning 1-2 months include evaluating Turkish-optimized embedding models such as multilingual-e5-large to address the performance gap on Turkish documents. Testing hybrid retrieval approaches that combine dense retrieval with BM25 could improve recall of relevant chunks. Adding a reranking layer using cross-encoder models could improve precision by better ordering the initially retrieved chunks.

Long-term enhancements over 3+ months involve more substantial changes. Training domain-specific embeddings on a Turkish academic corpus could dramatically improve retrieval quality for Turkish papers. Adding conversational memory would enable multi-turn interactions where the system maintains context across questions. Implementing citation graph retrieval would allow the system to understand relationships between papers and provide richer contextual understanding.

### 6.2 Evaluation Enhancements

The evaluation framework itself could be strengthened through several improvements. Immediately, expanding the test set to 10-15 papers would provide more robust statistics and better identify consistent patterns versus outliers. Calculating Krippendorff's alpha coefficient with a second human annotator would quantify inter-rater reliability and validate the evaluation consistency.

Long-term evaluation improvements include establishing a human baseline by having domain experts answer the same questions, allowing direct comparison of system performance to human performance. Comparing the system against commercial RAG solutions like ChatPDF or Claude's document analysis features would contextualize performance within the competitive landscape. Implementing automated retrieval metrics such as precision@k and NDCG would provide complementary quantitative measures. Finally, conducting user studies with actual researchers would assess real-world utility beyond artificial evaluation scenarios.

---

## 7. Conclusion

This work presents talk2DergiPark, a bilingual RAG system for Turkish academic papers, and conducts a structured qualitative evaluation of its performance. The system demonstrates strong performance on English source documents, achieving 4.92/5.0 with perfect fluency, relevance, and out-of-scope handling. Performance on Turkish source documents remains good at 4.41/5.0 but reveals opportunities for improvement in completeness and accuracy.

The evaluation methodology employing LLM-as-judge with human validation provides a scalable framework for iterative development. While limited to 2 papers in this proof-of-concept, the approach is extensible and demonstrates understanding of qualitative evaluation practices in NLP. The work makes several key contributions: a functional bilingual RAG system with browser integration, intelligent PDF processing with academic paper-specific optimizations, a systematic qualitative evaluation framework, and identification of language-specific performance gaps requiring further investigation.

The system is production-ready for simple factual queries but requires enhancement for complex analytical questions and improved Turkish document processing before deployment in research workflows. The identified performance gap between English and Turkish source documents provides a clear direction for future development efforts.

---

**Acknowledgments**: This project was developed for the Akbank GenAI Bootcamp. Thanks to DergiPark for maintaining Turkey's open-access academic platform.