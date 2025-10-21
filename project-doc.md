# Proof-of-Concept Evaluation: LLM-as-Judge Method

## Overview

This document describes the evaluation methodology used to test the talk2DergiPark RAG system. The evaluation uses ChatGPT-5 (Thinking) as an automated judge, with manual validation of results.

**Important**: This is a proof-of-concept using only 2 papers. 

---

## Why LLM-as-Judge?

Traditional NLP metrics like BLEU and ROUGE work well for translation but don't capture semantic quality in question-answering. They can't detect:
- Hallucinations 
- Incomplete answers
- Answers that miss the point
- Poor language quality

Using an LLM as a judge provides:
- Semantic understanding of answers
- Consistency across evaluations
- Scalability (can evaluate many responses quickly)
- Ability to assess nuanced quality dimensions

**Then I manually checked LLM's ratings**

---

## Evaluation Setup

### Test Cases
- 2 papers tested (1 English source, 1 Turkish source)
- 15 questions per paper = 30 total questions
- Question categories: objective, methodology, results, sample size, limitations, contributions, implications, out-of-scope

### Evaluation Dimensions (1-5 scale)
Each answer was rated on:
1. **Fluency** - Grammar and naturalness
2. **Relevance** - Does it answer the question?
3. **Accuracy** - Is the information correct?
4. **Completeness** - Enough detail?
5. **Language Appropriateness** - Correct language and terminology

<img width="1023" height="720" alt="Screenshot 2025-10-21 at 21 36 10" src="https://github.com/user-attachments/assets/b34e8ee4-57ce-40af-a4e1-2ffafb643d66" />

*Snippet of the rubrics* 



<img width="1027" height="493" alt="Screenshot 2025-10-21 at 22 05 23" src="https://github.com/user-attachments/assets/f62c3c81-332b-4d3a-81e1-955c2ad4d55c" />


---

## Process

1. **Generate responses**: RAG system answers all 30 questions
2. **LLM evaluation**: ChatGPT-5 rates each response on 5 dimensions
3. **Human validation**: Manually checked all ratings for accuracy
4. **Error analysis**: Classified specific problems using error taxonomy
5. **Statistical analysis**: Calculated averages, identified patterns

This hybrid approach combines automation (scalable, consistent) with human oversight (catches mistakes, validates quality).

---

## Results Summary

### Paper 1 (English Source)
- Overall: 4.92/5.0 (98.4%)
- Perfect fluency and relevance
- Only 2 errors total (1 omission, 1 overgeneralization)
- Bilingual performance: EN (4.97), TR (4.86)

### Paper 2 (Turkish Source)
- Overall: 4.41/5.0 (88.2%)
- Perfect fluency maintained
- 7 errors total (5 omissions, 2 format errors)
- Bilingual performance: TR (4.63), EN (4.23)


---

## What This Method Shows

**Strengths**:
- Zero hallucinations detected 
- Perfect out-of-scope handling 
- Good bilingual capability
- Consistent fluency across languages

**Weaknesses identified**:
- Completeness issues (especially on Turkish paper: 3.80/5.0)
- Accuracy drops on Turkish documents (4.87 → 4.00)
- Struggles with complex analytical questions
- Numerical extraction needs work

---

## Limitations

This is a **proof-of-concept** with limitations:
- Only 2 papers (need 20-50+ for real conclusions)
- Single validator (me) - no inter-rater reliability
- No comparison to other RAG systems or humans

The goal was to demonstrate the evaluation methodology, not provide definitive performance metrics.

---

## Tools Used

- **LLM Judge**: ChatGPT-5 (Thinking)
- **Validation**: Manual review of all ratings
- **Framework**: Custom 5-dimensional rubric + error taxonomy
- **Analysis**: Python script for statistical summaries
