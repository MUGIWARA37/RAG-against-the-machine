# RAG against the machine - Project Progress

## Key Definitions & Concepts

### RAG (Retrieval-Augmented Generation)
Large Language Models (LLMs) are frozen in time based on their training data and are prone to hallucinations when asked about niche, private, or recent topics. RAG solves this by turning a "closed-book exam" into an "open-book exam." Instead of relying on the LLM's internal weights to memorize facts, a RAG system first searches an external, verified database for the exact information needed, and then forces the LLM to generate an answer based *only* on those retrieved facts.

### Corpus
The foundational dataset or knowledge base that the system relies upon. In this project, the corpus is the raw source code and documentation of the `vLLM` repository. Processing a codebase corpus is uniquely challenging because it contains a mix of natural language (Markdown) and structured logic (Python), requiring specialized parsing to avoid destroying the semantic meaning of the text.

### Chunking & Overlap
LLMs have a strict limit on how much text they can process at once (the context window). Chunking breaks the massive corpus down into bite-sized segments (capped at 2000 characters per project specification).
* **Semantic Chunking:** Simply slicing a file every 2000 characters cuts Python functions or Markdown paragraphs in half. We use distinct strategies: `Language.MARKDOWN` for documentation and `Language.PYTHON` for code.
* **Chunk Overlap (e.g. 150 characters):** Eliminates boundary cuts where an answer span crosses the border between two adjacent chunks. Ensures complete sentences and definitions remain intact in at least one chunk.

### Inverted Index
Linear search through 14,000+ chunks for every question would take several minutes. An **inverted index** pre-indexes the corpus as a hash map:
`inverted_index: dict[str, list[tuple[doc_id, term_frequency]]]`
Instead of evaluating all 14,000 chunks, the search engine only checks documents containing the query words. This reduces retrieval latency to mere milliseconds, easily satisfying the requirement of $\le 90$ seconds for 200 questions.

### TF-IDF (Term Frequency - Inverse Document Frequency)
A foundational statistical scoring algorithm:
* **TF (Term Frequency):** Frequency of a word in a chunk.
* **IDF (Inverse Document Frequency):** Penalizes words that appear everywhere (e.g., "the", "def") and heavily boosts rare, topic-specific keywords.
**Formula:** `TF-IDF(t, d, D) = TF(t, d) * log(N / |{d ∈ D : t ∈ d}|)`

### BM25 (Best Matching 25) — From Scratch Implementation
The industry standard lexical ranking algorithm, correcting TF-IDF's flaws. As required by school peer evaluations, BM25 must be implemented in pure Python from first principles rather than relying on external blackbox libraries (`rank-bm25`).

#### The Core Mechanics:
1. **Term Saturation ($k_1$ parameter, default 1.5):** In TF-IDF, term frequency scales linearly without bound. BM25 recognizes that seeing a keyword 5 times is better than 1 time, but seeing it 50 times is not 10x better. The score saturates asymptotically.
2. **Length Normalization ($b$ parameter, default 0.75):** A long 2000-character chunk naturally contains more words than a concise 200-character chunk. BM25 penalizes excessively long chunks and rewards concise matches relative to the average document length ($\text{avgdl}$).
3. **IDF (Robertson-Spärck Jones formula):**
   $$\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$$
   Where $N$ is total chunks and $n(q_i)$ is how many chunks contain term $q_i$.

#### BM25 Scoring Formula:
$$\text{Score}(q, d) = \sum_{q_i \in q} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

### Code-Aware Tokenization & Path Weighting
Lexical search fails when queries and code use different naming conventions:
* **`snake_case` Splitting:** `fused_batched_moe` splits into `["fused_batched_moe", "fused", "batched", "moe"]`.
* **`camelCase` / `PascalCase` Splitting:** `AsyncLLMEngine` splits into `["asyncllmengine", "async", "llm", "engine"]`.
* **Punctuation Stripping:** Removes code syntax like `()`, `{}`, `[]`, `:`, `.`, `"` so words aren't glued to punctuation.
* **File Path Inclusion:** File paths (e.g., `vllm/attention/ops/triton_flash_attention.py` or `docs/features/lora.md`) contain high-signal keywords. Indexing the path alongside the text lets BM25 instantly find the correct module.

### Recall@k & 1D IoU (Intersection over Union)
Official metric used to measure retrieval quality:
* **IoU > 0.05:** Retrieved snippet and ground-truth must belong to the exact same file and overlap by at least 5%:
  $$\text{IoU} = \frac{\text{Intersection}}{\text{Union}} = \frac{\max(0, \min(end_1, end_2) - \max(start_1, start_2))}{\text{len}_1 + \text{len}_2 - \text{Intersection}}$$
* **Recall@k:** Proportion of correct reference sources retrieved within the top-$k$ results ($k \in [1, 3, 5, 10]$).
* **Thresholds:** $\ge 80\%$ Recall@5 on documentation, and $\ge 50\%$ Recall@5 on code questions.

### Generation & Augmentation
Passes the retrieved top-$k$ context snippets into `Qwen/Qwen3-0.6B` using a strict prompt enforcing answers grounded solely in the retrieved context.

### Pydantic Models & Python Fire
* **Pydantic:** Strictly validates schemas for `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`.
* **Python Fire:** Automatically maps Python methods in `Cli` to CLI flags and subcommands.

---

## Currently Implemented

- **Data Models** (`src/models.py`): Complete Pydantic schemas.
- **Chunking Strategy** (`src/indexing/chunking.py`): Distinct Markdown and Python splitters.
- **Index Building** (`src/indexing/builder.py`): Corpus crawling and serialization.
- **Retrieval Engine** (`src/retrieval/searcher.py`): Search execution over index.
- **LLM Generation** (`src/generation/llm.py`): `Small_llm` wrapper around `Qwen/Qwen3-0.6B`.
- **Makefile** (`Makefile`): Mandatory targets `install`, `run`, `debug`, `clean`, `lint`, and `lint-strict`.
- **CLI Commands** (`src/__main__.py` & `src/cli.py`):
  - `index`: Corpus indexing.
  - `search`: Single-query retrieval.
  - `search_dataset`: Batch retrieval writing `StudentSearchResults` JSON.
  - `answer`: Single-query search + generation.
  - `answer_dataset`: Batch generation with `tqdm` progress tracking writing `StudentSearchResultsAndAnswer` JSON.
  - `evaluate`: Local evaluation command reporting Recall@1, 3, 5, 10.
- **Local Evaluator** (`src/retrieval/evaluator.py`):
  - Fully written and verified by user: `compute_iou`, `is_source_match`, and `evaluate_search_results`.
  - Tested on `dataset_docs_public.json`.
- **Environment & Storage**: Large wheels and virtualenv safely redirected to `/goinfre`.

---

## Current Task: Custom BM25 & Tokenizer from Scratch

- **Objective**: Replace external library `rank-bm25` with our own custom `BM25` engine and code-aware tokenizer.
- **Performance Targets**:
  - Documentation Recall@5: $\ge 80\%$ (simulated: **87.0%**).
  - Code Recall@5: $\ge 50\%$ (simulated: **69.7%**).
  - Retrieval throughput: $\le 90$s for 200 questions.

---

## Next Steps Planned

1. Build custom `BM25` class with inverted index and Robertson-Spärck Jones IDF in `src/indexing/bm25.py`.
2. Build code-aware tokenizer (`snake_case`, `camelCase`, punctuation stripping) in `src/indexing/tokenizer.py`.
3. Integrate custom BM25 and tokenizer into `builder.py` and `searcher.py`.
4. Make `max_chunk_size` dynamic via CLI argument throughout chunkers.
5. Run `make lint` (`flake8` and `mypy` strict checks).
6. Complete final `README.md` following 42 subject guidelines.
