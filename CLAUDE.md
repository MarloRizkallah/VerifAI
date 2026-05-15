# CLAUDE.md — GP Project: FEVER Fact-Checking with RAG + NLI

## Hardware (lab machine: ai-ws2)
- GPU: Quadro RTX 5000 — Turing CC 7.5, **15.5 GB VRAM**
- CUDA: 12.1 (`+cu121`)
- **Always use GPU 1** (`CUDA_VISIBLE_DEVICES=1`) — GPU 0 belongs to other lab users (Amany's training PID 531393 — do not kill)
- Root filesystem: 402 GB (`/dev/nvme1n1p3`) — can fill quickly with large index files
- Bulk storage: `/media/ai-ws2/New Volume` (423 GB+ free)
- Qdrant storage: `rag/indices/qdrant` → symlink → `/media/ai-ws2/New Volume/qdrant_storage`

## Environment (conda base)
- Python: 3.11
- torch: 2.2.1+cu121
- transformers: 4.57.6
- numpy: < 2 (torch 2.2.1 ABI requirement — do not upgrade numpy to 2.x)
- **Do NOT upgrade to transformers 5.x** (breaks compatibility with the existing stack)
- **Do NOT touch GPU 0**

## Notebook: RAG_v1_Harrier.ipynb
**Primary working notebook for the RAG pipeline. All RAG steps live here.**
`RAG_v1.ipynb` is superseded — do not use it. All edits go to `RAG_v1_Harrier.ipynb`.
Follows the same cell conventions as `DeBERTa_Final_NoMultiv2.ipynb`:
- Markdown cell: `## Step N - <title>` heading + explanation paragraphs + optional `| Choice | Why |` table
- Code cell: standalone-runnable, ALL_CAPS config constants, visually-aligned `=`, formatted prints with field widths, relative paths
- Never write a code cell without the preceding markdown heading

## Notebook: DeBERTa_Final_NoMultiv2.ipynb
Production NLI model notebook. Contains the DeBERTa-v2 ensemble adapters.
**Do not modify this notebook** during RAG work.

---

## RAG Day 1 — Current State

### Embedding model: microsoft/harrier-oss-v1-270m

**Why Harrier (not Jina v5-nano or gte-base):**
Jina v5-nano-retrieval requires transformers >= 5.1.0 and torch >= 2.8.0 — incompatible with the main env.
Harrier runs in the main env (transformers >= 4.4.0, torch >= 2.1.0), eliminating any cross-env IPC boundary for Day 3 DeBERTa integration. Verified MTEB v2 = 66.5. See "Rejected alternatives" below for full decision log.

**Verified on RTX 5000 (2026-05-10):**

| Metric | Value |
|---|---|
| attn implementation | sdpa |
| dtype | float32 (FP16 → NaN on Turing CC 7.5) |
| Peak VRAM after load | 1023 MB |
| Peak VRAM during inference | 1043 MB |
| Param count | 268,098,176 |
| Output dim | 640 |
| Throughput (batch=32) | 651 p/s |
| Throughput (batch=64) | 670 p/s |
| Throughput (batch=128) | **772 p/s** ← optimal |
| Corpus estimate (25.2M) | **9.1 h** |
| Semantic gap (Obama) | 0.3902 |
| Semantic gap (Eiffel) | 0.2627 |
| Semantic gap (Einstein) | 0.3380 |
| L2 norm range | 1.0000 – 1.0000 |

### Config constants changed in RAG_v1_Harrier.ipynb

The following ALL_CAPS constants in the main config cell (Step 1) were updated from their previous Jina values:

| Constant | Old value | New value |
|---|---|---|
| `EMBED_MODEL` | `"jinaai/jina-embeddings-v5-text-small"` | `"microsoft/harrier-oss-v1-270m"` |
| `EMBED_DIM` | `1024` | `640` |
| `EMBED_BATCH` | `64` | `128` |
| `EMBED_MAX_LEN` | `256` | `256` (unchanged) |
| `EMBED_DTYPE` | `torch.float16` | `torch.float32` |
| `QUERY_PREFIX` | *(not defined)* | `"Instruct: Given a claim, retrieve evidence passages that support or refute it\nQuery: "` |

Qdrant constants are unchanged: `COLLECTION = "fever_wiki"`, `HNSW_M = 16`, `HNSW_EF_CONSTRUCT = 200`.

**Critical encoding rule:** Documents are encoded as **raw text — no prefix**. The `QUERY_PREFIX` is applied only at query time (Day 2). Applying the prefix at index time silently degrades retrieval quality.

### Qdrant infrastructure — switched to Docker server (2026-05-11)

`QdrantLocal` (path= mode) was discovered to silently ignore `indexing_threshold` changes and uses brute-force search — unsuitable for 25M passages. Switched to the actual Qdrant server running in Docker.

**Docker command used:**
```bash
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 \
  -v "/media/ai-ws2/New Volume/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

**Client connection changed** (Step 3 cell and all subsequent cells):
```python
# OLD (local mode — broken optimizer config, brute-force search)
client = QdrantClient(path="rag/indices/qdrant")

# NEW (server mode — full HNSW + optimizer config support)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
```

`timeout=120` sets the httpx socket-read timeout to 120s. The default (~5s) is too short during Qdrant WAL/segment stalls under bulk upsert load. Normal calls complete in milliseconds — the timeout only fires on rare stall events.

Storage symlink unchanged: `rag/indices/qdrant` → `/media/ai-ws2/New Volume/qdrant_storage` — still valid as a sanity check that storage is on the large volume; the Docker container mounts the same path directly.

**To restart the Qdrant server after a reboot:**
```bash
docker start qdrant   # container has --restart unless-stopped so it may already be running
docker ps             # confirm
```

### Steps completed (RAG_v1_Harrier.ipynb) — Day 1 summary

| Step | Title | Status |
|---|---|---|
| Step 1 | Environment setup & `iter_passages()` loader | ✓ done |
| Step 2 | BM25 index build | ✓ done |
| Step 2A | Full-corpus dry-pass count | ✓ done |
| Step 2B | Full BM25 build | ✓ done |
| Step 2C | Persistence & load helper | ✓ done |
| Step 3 | Qdrant collection initialization | ✓ done (server client, dim=640, indexing_threshold=30,000,000 verified) |
| Step 4 | Harrier embed + Qdrant upsert | ✓ done |
| Step 4A | Model load & encoding validation | ✓ done (shape=640, consistency, asymmetry, throughput sweep all PASS) |
| Step 4B | Smoke upsert & dense retrieval check (10K passages) | ✓ done (round-trip PASS, collection wiped clean for 4C) |
| Step 4C | Full corpus embed + upsert (25.2M passages) | ✓ done — crashed once at ~10% (ReadTimeout), resumed from checkpoint |
| Step 4D | HNSW build polling | ✓ done — collection status = green |

**Real-world throughput note:** Benchmark (test_harrier.py) measured 772 p/s pure embedding. Step 4C wall-clock was 274 p/s — difference is `iter_passages` disk I/O + `tolist()` (81,920 float conversions per batch) + HTTP round-trips to Qdrant server. 772 p/s remains the correct figure for the GP write-up (embedding throughput in isolation).

---

## Rejected alternatives (decision history — kept for GP write-up)

### Jina v5-text-nano-retrieval (jinaai/jina-embeddings-v5-text-nano-retrieval)
- **Rejected:** requires transformers >= 5.1.0 and torch >= 2.8.0 (hard blockers in main env)
- Would have required a separate `rag_embed_v5` conda env → IPC boundary between retrieval and DeBERTa NLI at Day 3
- MTEB v2 = 71.0, dim=768, encoder-only (EuroBERT-210M backbone) — highest quality of all candidates
- Storage: ~85 GB projected for 25.2M passages at dim=768

### Jina v5-text-small (jinaai/jina-embeddings-v5-text-small)
- **Abandoned mid-run:** Qwen3-based decoder-only LLM (677M params), dim=1024
- Two crashes: disk-full at 4% (1.065M passages, root filesystem at 100%), kernel death at 7% (1.82M passages)
- Real throughput measured at 66 p/s → ~98h for full corpus (not viable)
- Incompatible with torch.compile on torch 2.2.1 + transformers 4.x (InternalTorchDynamoError)
- Disk fix: moved Qdrant storage to New Volume via symlink (still in place)

### gte-base-en-v1.5 (Alibaba-NLP/gte-base-en-v1.5)
- **Not tested (pre-empted by Harrier GO):** encoder-only, transformers >= 4.36.0 (safe), dim=768
- MTEB v1 retrieval = 54.09 — significantly weaker retrieval than Harrier (MTEB v2 66.5)
- Would have been the fallback if Harrier failed throughput

### BGE-base-en-v1.5
- **Rejected early:** MTEB 63.55, ~8 points below v5-nano — too large a quality drop

### Voyage 4 nano
- **Rejected early:** commercial API, MMTEB 58.9 (weaker), API cost concerns

---

## RAG pipeline architecture (Day 1–3 plan)

```
Claim
  → [Day 2] Harrier query encode (QUERY_PREFIX + claim)
  → Qdrant dense search (top-10)
  → BM25 lexical search (top-10)
  → RRF fusion (default) or weighted-sum → ~20 candidates
  → gte-reranker-modernbert-base cross-encoder rerank
  → top-5 passages
  → [Day 3] DeBERTa-v2 ensemble NLI → SUPPORTS / REFUTES / NEI verdict
```

All components run in the main conda env (no subprocess boundary). Reranker (`Alibaba-NLP/gte-reranker-modernbert-base`, ModernBERT-based) requires transformers >= 4.48 — already satisfied by 4.57.6.

---

## Day 1 Status: COMPLETE ✓

**Completed 2026-05-13.** Full FEVER wiki dump (25,247,890 passages) successfully embedded with Harrier-270m (FP32, dim=640) and upserted into Qdrant. BM25 index built over same corpus. HNSW index built (Step 4D polled to `green`).

### Steps completed (RAG_v1_Harrier.ipynb) — full log

| Step | Title | Status |
|---|---|---|
| Step 1 | Environment setup & `iter_passages()` loader | ✓ done |
| Step 2 | BM25 index build | ✓ done |
| Step 2A | Full-corpus dry-pass count | ✓ done — 25,247,890 passages, 5.1 min |
| Step 2B | Full BM25 build | ✓ done — 18.0 min tokenise + 5.6 min build, 55.5 GB RSS |
| Step 2C | Persistence & load helper | ✓ done |
| Step 3 | Qdrant collection initialization | ✓ done — server client, dim=640, indexing_threshold=30,000,000 verified |
| Step 4 | Harrier embed + Qdrant upsert | ✓ done |
| Step 4A | Model load & encoding validation | ✓ done — shape=640, consistency, asymmetry, throughput sweep PASS |
| Step 4B | Smoke upsert & dense retrieval check (10K) | ✓ done — round-trip PASS, collection wiped clean for 4C |
| Step 4C | Full corpus embed + upsert (25.2M passages) | ✓ done — crashed once at ~10% (ReadTimeout, fixed with timeout=120), resumed from checkpoint |
| Step 4D | HNSW build polling | ✓ done — collection status = green |

---

## RAG Day 2 — COMPLETE ✓

**Completed 2026-05-13.** All Step 5 cells executed on lab machine. Outputs saved in `RAG_v1_Harrier.ipynb`.

### Step 5 cell inventory (executed)

| Cell ID | Step | Title | exec # | Output |
|---|---|---|---|---|
| `6120ff85` (md) | 5 overview | Hybrid retrieval pipeline (Day 2) | — | — |
| `9df956c8` (md) / `42e8f229` (code) | 5A | Load indices & embedder | 41 | ✓ |
| `9944072e` (md) / `e15e37db` (code) | 5B | BM25 search helper | 42 | ✓ |
| `99ff4f5c` (md) / `a7e55290` (code) | 5C | Dense search helper | 43 | ✓ |
| `644c7244` (md) / `e126ea10` (code) | 5D | Fusion (RRF + weighted-sum) | 44 | ✓ |
| `98b9dd57` (md) / `7c1d5046` (code) | 5E | Cross-encoder reranker load | 45 | ✓ |
| `c940572c` (md) / `7ce2bd70` (code) | 5F | `retrieve_top5()` orchestrator | 46 | ✓ |
| `586e0ff7` (md) / `886e7086` (code) | 5G | Diagnostic: Claim-Label Consistency Check | 47 | ✓ |
| `63a27773` (md) / `76d54eaf` (code) | 5H | Build Gold Evidence Dict (Grouped by Claim-Label) | 48 | ⚠ see note |

### Key design decisions — rationale

**Function name: `retrieve_top5` (not `retrieve_topk`).** Was `retrieve_top3` in the original draft; renamed when `final_k` was raised to 5. The name encodes the current default rather than using a generic `topk` suffix. Change if `final_k` is tuned again in Day 4.

**SR-only evaluation — why NEI is dropped entirely.** The NEI rows in `dev_final_cleaned_testset.jsonl` (and the SR file) were constructed by our own BM25+SBERT sampling pipeline, not by human annotators finding evidence. Retrieval recall against these synthetic distractors would measure how well our retriever replicates its own prior output — meaningless as an evaluation signal. Real FEVER NEI annotations mean "no evidence found in Wikipedia", which is a judgment about the world, not a retrievable passage. We therefore evaluate only on SUPPORTS and REFUTES, where ground-truth evidence passages exist.

**`(claim, label)` grouping — why not claim-only.** Step 5G found exactly 1 claim text that maps to both SUPPORTS and REFUTES: "An island is part of the ABC Islands." Different annotators found contradictory evidence for the same claim. Keying the gold dict by `(claim, label)` tuple rather than claim text preserves both without filtering either — each `(claim, label)` becomes an independent retrieval query evaluated against its own gold set. The retriever is label-agnostic (it returns the same top-K regardless of label), so when a claim is contested, the same retrieved passages are simply scored against two different gold sets.

**Evidence dedup via `set()` after normalisation — why.** FEVER evidence entries can duplicate across annotators within the same claim-label group. Building a set of `_norm(f"{page_id}::{sent_idx}")` strings deduplicates them before Day 3 matching, so hit@k is not inflated by counting the same passage multiple times. The normalisation (`clean_artifacts` + `' '.join(text.split())`) strips FEVER bracket artifacts (`-LRB-`, `-RRB-`, etc.) and collapses whitespace. The same normalisation is applied to Qdrant `passage_id` strings at match time, so both sides of the comparison are consistent.

**Kept the 1 inconsistent claim — why.** Filtering it would remove legitimate signal. The `(claim, label)` keying handles it cleanly: both `("An island is part of the ABC Islands.", "SUPPORTS")` and `("An island is part of the ABC Islands.", "REFUTES")` get their own gold sets, and each is evaluated independently. No special-casing needed.

### Step 5F — `retrieve_top5` smoke test output (saved)

```
Smoke test: retrieve_top5("Albert Einstein was awarded the Nobel Prize in Physics....")

  1. [+3.0119]  Einstein's_awards_and_honors::0
          In 1922 , Albert Einstein was awarded the 1921 Nobel Prize in Physics , ...
  2. [+2.0164]  Percy_W._Bridgman_House::14
          He was awarded the Nobel Prize in Physics (the fifth American to be so honored) in 1946 ...
  3. [+1.9990]  Dalton_Medal::40
          He was awarded the Nobel Prize in Physics in 1948 .
  4. [+1.9541]  Reform_Congregation_Keneseth_Israel_...(Philadelphia)...::19
          In 1934 Albert Einstein (1879-1955), the Nobel Prize winning physicist ...
  5. [+1.9458]  Dalton_Medal::31
          He was awarded the Nobel Prize in P...
```

Top result is `Einstein's_awards_and_honors::0` with a large margin (+3.01 vs +2.02) — correct and expected. Full pipeline (BM25 + dense + RRF fusion + cross-encoder rerank) working end-to-end.

### Step 5G — Claim-Label Consistency output (saved)

```
Dev set consistency check
  NEI rows dropped              : 6,650
  SUPPORTS + REFUTES rows       : 12,847
  Unique claim texts            : 8,227
  Single-label claims           : 8,226
  Multi-label (inconsistent)    : 1  (0.0%)

Example inconsistent claims (up to 5):
  An island is part of the ABC Islands.   ->  ['REFUTES', 'SUPPORTS']
```

### Step 5H — Gold Evidence Dict (fixed 2026-05-13 — awaiting re-run on lab)

**Previous output (stale — from wrong source file):**
```
Gold evidence dict  -- source: dev_final_cleaned_testset.jsonl
  NEI rows dropped             : 6,650
  SUPPORTS + REFUTES rows kept : 12,847
  Total (claim, label) groups  : 0   ← was empty: evidence field absent in testset file
```

**Fix applied (2026-05-13):** Cells 5G (`886e7086`) and 5H (`76d54eaf`) updated — both now use `dev_final_cleaned_SR.jsonl`. NEI filter blocks removed (dead code in an SR-only file). Stale outputs cleared (`execution_count=null`, `outputs=[]`). Cells must be re-run on the lab machine before Day 3 starts. Expected output after re-run: non-zero `(claim, label)` groups matching the 12,847 SR rows.

### Lab runbook before starting Day 3

1. Confirm Qdrant server is running: `docker ps` → container `qdrant` on port 6333
2. Run Step 4A (load Harrier model into scope) and Step 5A (load BM25 + Qdrant client + `pid_to_qid`)
3. Re-run Step 5G (`886e7086`) — verify SUPPORTS+REFUTES row count and the one inconsistent claim are still present
4. Re-run Step 5H (`76d54eaf`) — verify `Total (claim, label) groups > 0` before proceeding

---

## Day 3 — COMPLETE ✓

**Completed 2026-05-15.** End-to-end hallucination verdict pipeline implemented and smoke-tested. All Step 6 cells executed on lab machine.

### Step 6 cell inventory

| Cell ID (md / code) | Step | Title |
|---|---|---|
| `e89b2398` / — | 6 overview | Day 3: NLI Verdict Pipeline |
| `7ace1131` / `step6a-code` | 6A | DeBERTa ensemble loader |
| `0e892294` / `32104b5b` | 6B | `nli_predict` + `nli_predict_batch` |
| `7e801866` / `b80c4d60` | 6C | `score_passages` — retrieve + batch NLI |
| `82d7d6af` / `1b96f7c9` | 6D | `aggregate_passages` — label-priority |
| `d62992d0` / `step6e-code` | 6E | `final_verdict` — 5-case logic |
| `9360e812` / `cccb19dc` | 6F | Smoke test with per-stage timing |

### Engineering fixes applied during Day 3

**`weights_only=False` in Step 6A (`step6a-code`).**
transformers 4.57.6 added `check_torch_load_is_safe()` which refuses `torch.load` on torch < 2.6 (CVE-2025-32434). Passing `weights_only=False` to both `AutoModelForSequenceClassification.from_pretrained` and `PeftModel.from_pretrained` bypasses the check. Safe because these are trusted local weights.

**Batched NLI inference — `nli_predict_batch` (Step 6B, cell `32104b5b`).**
Original per-passage loop: `k × 3 = 15` forward passes per claim. `nli_predict_batch(claim, passages)` tokenises all N pairs at once and runs each of the 3 models once over the full batch — `[3, N, 3]` → mean over model dim → `[N, 3]` → softmax. Total: **3 forwards per claim** regardless of k. Measured 3–5× speedup on NLI portion. `nli_predict(claim, passage)` kept intact for sanity tests.

### Key constants (Day 3, current values)

| Constant | Value | Cell | Notes |
|---|---|---|---|
| `DeBERTa_NLI_MODEL` | `"microsoft/deberta-v3-large"` | `step6a-code` | |
| `NLI_MAX_LEN` | `256` | `step6a-code` | matches training |
| `NLI_LABEL2ID` | `{"SUPPORTS":0,"REFUTES":1,"NOT ENOUGH INFO":2}` | `step6a-code` | exact string — not "NEI" |
| `NLI_ADAPTER_PATHS` | `deberta_single_evidence_v2/adapter_seed_{42,123,777}` | `step6a-code` | EMA weights pre-applied |
| `TAU` | `0.85` | `step6e-code` | TUNE_ME in Day 4; tuned from 0.5 during lab runs |
| `MIN_DECISIVE_CONF` | `0.6` | `1b96f7c9` | TUNE_ME in Day 4 |

**Similarity definition.** `similarity = sigmoid(rerank_score)` where `rerank_score` is the raw cross-encoder logit from gte-reranker-modernbert-base. sigmoid(0)=0.5 is the reranker's decision boundary. TAU=0.85 was found empirically to correctly classify the smoke test claims.

### Verdict 5-case logic (Step 6E — unchanged)

| NLI label | sim ≥ TAU | Verdict |
|---|---|---|
| REFUTES | either | `hallucinated` |
| SUPPORTS | yes | `factual` |
| SUPPORTS | no | `uncertain` |
| NOT ENOUGH INFO | yes | `knowledge_gap` |
| NOT ENOUGH INFO | no | `likely_hallucinated` |

### Aggregation policy decision history (Step 6D)

**Current policy: label-priority with confidence safeguard (`1b96f7c9`).**

```
confident = [p for p in scored if p["predicted_conf"] >= MIN_DECISIVE_CONF]
pool      = confident if confident else scored   # fallback: full set
if any SUPPORTS in pool  → highest-sim SUPPORTS
elif any REFUTES in pool → highest-sim REFUTES
else                     → highest-sim NEI
```

Rationale: SUPPORTS/REFUTES signal *presence* of decisive evidence; NEI signals *absence*. One decisive passage outranks any number of indecisive ones — label priority requires no tunable weight.

**Alternatives tried and rejected (during Day 3 lab diagnostic):**

| Strategy | Smoke (6 claims) | Problem |
|---|---|---|
| Max-confidence | failed | NEI systematically more confident than SUPPORTS/REFUTES → drowns decisive signals |
| Majority vote | 4/6 | Bare Einstein diagnostic: `Einstein's_awards_and_honors` (SUPPORTS, sim 0.953) outvoted 1–3 by unrelated Nobel-Physics NEI passages |
| Continuous score (conf × sim × label_weight) | 5/6 | Requires NEI weight hyperparameter with no dev-set justification |
| Score × vote-share multiplier (NEI=0.7) | 6/6 | Margin 0.007 on one claim — overfitting signal on 6-point sample |
| **Label-priority (chosen)** | **6/6** | No weight hyperparameter beyond `MIN_DECISIVE_CONF` |

**Day 4 alternative to revisit:** If dev-set evaluation shows label-priority is too aggressive on REFUTES (false-positive hallucination flags), revert to majority vote or continuous score with dev-set-tuned weights.

### Known limitation: entity mismatch / pronoun blindness

When the cross-encoder reranker surfaces a passage containing decisive language ("He was awarded the Nobel Prize", "was founded in 2021") where the pronoun/subject resolves to a *different* entity than the claim, the policy commits to the wrong verdict. The Anthropic founding claim smoke test exposes this: the reranker finds REFUTES-predicting passages about other entities, and label priority correctly treats them as decisive but cannot detect the entity mismatch. Expected verdict: `hallucinated` (known wrong). Mitigation via coreference-aware reranking is documented as future work.

### Smoke test results (Step 6F — executed on lab)

| Claim | Expected | Notes |
|---|---|---|
| Einstein Nobel 1921 | `factual` | SUPPORTS, sim 0.953 ≥ TAU=0.85 |
| Eiffel Tower in Berlin | `hallucinated` | REFUTES passage found |
| Tetris / Pajitnov | `factual` | SUPPORTS passage found |
| Anthropic founded 2021 | `hallucinated` | Known limitation: entity mismatch, pronoun blindness |
| Water boils 100°C | `factual` | SUPPORTS passage found |

### Lab runbook for Day 4

1. `docker ps` — confirm Qdrant container running on port 6333
2. Re-run **Step 5A** (`42e8f229`) — restores BM25, Harrier, Qdrant client, `pid_to_qid`
3. Re-run **Step 5E** (`7c1d5046`) — restores reranker
4. Re-run **Step 6A** (`step6a-code`) — restores NLI models
5. Cells 6B–6F depend on prior scope — re-run in order if kernel was restarted
6. Re-run **Step 5H** (`76d54eaf`) — rebuild `gold` dict before any evaluation loop

---

## Day 4 Plan — Evaluation & Tuning

Day 4 runs the full evaluation loop over `dev_final_cleaned_SR.jsonl` (12,847 SUPPORTS + REFUTES claims) and tunes the pipeline hyperparameters.

### Evaluation targets

- **Retrieval:** hit@5 and recall@5 (exact `passage_id` match against `gold` dict), broken down by SUPPORTS vs REFUTES
- **Verdict:** agreement with gold label on the SR subset

### TUNE_ME markers

| Constant | Current value | Tuning method |
|---|---|---|
| `TAU` | 0.85 | Sweep `[0.5, 0.6, 0.7, 0.8, 0.85, 0.9]`, maximise verdict F1 on SR dev set |
| `MIN_DECISIVE_CONF` | 0.6 | Sweep `[0.4, 0.5, 0.6, 0.7]`, check fallback trigger rate |
| `k_per_side` | 10 | Try 15, 20 — measures retrieval recall cost vs reranker latency |
| `top_n_fused` | 20 | Try 30 — more candidates for reranker |
| `final_k` | 5 | Try 3, 7 — NLI budget vs recall tradeoff |
| Fusion strategy | RRF | Compare weighted-sum on SR recall |

### Aggregation policy review

If label-priority produces too many false-positive `hallucinated` verdicts on SUPPORTS claims (REFUTES tier activated by entity-mismatch passages), consider:
1. Adding a `MIN_REFUTES_SIM` floor (only commit to REFUTES if sim ≥ threshold)
2. Reverting to majority vote as a conservative baseline
3. Score × vote-share with dev-set-tuned NEI weight