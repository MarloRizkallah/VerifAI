# VerifAI — AI Hallucination Detection & Verification System

> A hybrid AI system that detects hallucinations in LLM-generated text by extracting factual claims, retrieving Wikipedia evidence via RAG, and classifying each claim with a fine-tuned NLI model.

---

## NLI Model Performance

Fine-tuned DeBERTa-v3-large + LoRA ensemble on 157,089 FEVER claim–evidence pairs (3 seeds).

| Split | Macro-F1 | Accuracy |
|---|---|---|
| Validation (ensemble) | 0.9680 | — |
| **Test (held-out)** | **0.9547** | **0.9548** |

*Note: these metrics reflect per-pair NLI classifier performance, not end-to-end hallucination detection.*

---

## Repository Structure

### `GP.ipynb` — Data Preparation
Builds the training and evaluation datasets from the FEVER corpus. Covers claim–evidence pair construction, NEI negative mining (BM25 + SBERT hybrid), artifact cleaning, deduplication, and class balance analysis. Produces `final_cleaned_nli_dataset.jsonl` (157K pairs) and `dev_final_cleaned_testset.jsonl` (19.5K pairs).

### `DeBERTa_Final_NoMulti_v3.ipynb` — Model Training & Evaluation
Full training pipeline for the DeBERTa-v3-large + LoRA NLI classifier. Includes LoRA config, R-Drop, FGM adversarial training, EMA, label smoothing, and the ensemble inference logic. The notebook reflects the v4 training run (for complete outputs), but the production model is **v2** — selected for its superior generalisation (Test F1 0.9547 vs. a larger val–test gap in v4).

### `claim_extractor.py` — Claim Extraction
Hybrid claim extractor: a heuristic prefilter identifies candidate sentences, then an LLM (Ollama llama3.1:8b or Anthropic) decomposes them into atomic, self-contained factual claims with pronouns resolved. Supports `ollama`, `openai`, and `anthropic` backends.

### `RAG_v2_Harrier.ipynb` — RAG Pipeline & Full Verification
End-to-end retrieval-augmented verification pipeline. Covers:
- **Dense retrieval:** Harrier-270m embeddings → Qdrant (~25.2M Wikipedia sentences)
- **Sparse retrieval:** vectorised BM25 (bm25s — 100–500× faster than standard BM25)
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranking:** gte-reranker-modernbert-base cross-encoder
- **NLI classification:** batched DeBERTa ensemble inference
- **5-case verdict logic:** SUPPORTS/REFUTES/NEI × similarity threshold → factual / hallucinated / uncertain

### `backend/` + `frontend/` — Web Application (VerifAI)
Full-stack web app. FastAPI backend with SSE streaming serves the complete pipeline as an API. Next.js frontend renders a highlighted paragraph with per-claim verdict cards, confidence bars, Wikipedia evidence snippets, and LLM-generated explanations for flagged claims.

---

## Tech Stack

| Component | Technology |
|---|---|
| NLI model | DeBERTa-v3-large + LoRA (PEFT), fine-tuned on FEVER |
| Dense retrieval | Harrier-270m → Qdrant |
| Sparse retrieval | bm25s (vectorised BM25) |
| Reranker | gte-reranker-modernbert-base |
| Claim extraction / explanation | Ollama llama3.1:8b |
| Backend | FastAPI + sse-starlette |
| Frontend | Next.js 14, React 19, Tailwind CSS |
| Knowledge base | FEVER Wikipedia dump (~25.2M sentences) |

---

## Acknowledgements

FEVER dataset (Thorne et al., 2018) · DeBERTa-v3 (He et al., 2021) · Harrier embedder (Lightblue AI) · gte-reranker-modernbert-base (Alibaba DAMO) · bm25s (Lù, 2024) 

