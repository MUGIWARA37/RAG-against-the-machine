# RAG against the machine - Project Progress

## Key Definitions & Concepts

### RAG (Retrieval-Augmented Generation)
Large Language Models (LLMs) are frozen in time based on their training data and are prone to hallucinations when asked about niche, private, or recent topics. RAG solves this by turning a "closed-book exam" into an "open-book exam." Instead of relying on the LLM's internal weights to memorize facts, a RAG system first searches an external, verified database for the exact information needed, and then forces the LLM to generate an answer based *only* on those retrieved facts.

### Corpus
The foundational dataset or knowledge base that the system relies upon. In this project, the corpus is the raw source code and documentation of the `vLLM` repository. Processing a codebase corpus is uniquely challenging because it contains a mix of natural language (Markdown) and structured logic (Python), requiring specialized parsing to avoid destroying the semantic meaning of the text.

### Chunking
LLMs have a strict limit on how much text they can process at once (the context window). Chunking is the process of breaking the massive corpus down into bite-sized segments (in this project, capped at 2000 characters). 
* **Semantic Chunking:** Simply slicing a file every 2000 characters might cut a Python function or a Markdown paragraph in half, destroying its meaning. Therefore, chunking strategies use headers (for Markdown) or structural syntax (for Python) to ensure chunks remain coherent and readable.

### Indexing
Searching through thousands of files linearly for every question would take minutes. Indexing solves this by processing the chunks ahead of time into a highly optimized data structure (like an inverted index). This allows the system to cross-reference query keywords against the entire corpus and return results in milliseconds.

### TF-IDF (Term Frequency - Inverse Document Frequency)
A foundational algorithm for scoring how relevant a specific word is to a document. 
* **TF (Term Frequency):** The more times a word appears in a chunk, the more relevant that chunk is to the word. 
* **IDF (Inverse Document Frequency):** Common words (like "the" or "def") appear in almost every chunk, making them useless for searching. IDF scales down the score of words that appear everywhere and heavily boosts rare, highly specific keywords.
**Formula:** `TF-IDF(t, d, D) = TF(t, d) * log(N / |{d ∈ D : t ∈ d}|)`

### BM25 (Best Matching 25)
The industry standard for lexical search, expanding on TF-IDF to solve two major flaws:
1. **Term Saturation (`k_1` parameter):** In TF-IDF, if a document contains a keyword 100 times, its score explodes. BM25 recognizes that seeing a keyword 5 times is better than 1 time, but seeing it 100 times isn't significantly better than 10 times. The score mathematically caps out (saturates).
2. **Length Normalization (`b` parameter):** A 2000-character chunk naturally contains more words than a 200-character chunk, giving it an unfair advantage in TF-IDF. BM25 penalizes excessively long documents and rewards concise documents based on the average document length (`avgdl`).
**Formula:** `Score(q, d) = ∑ [ IDF(q_i) * (f(q_i, d) * (k_1 + 1)) / (f(q_i, d) + k_1 * (1 - b + b * (|d| / avgdl))) ]`

### Retrieval
The active phase where a user's natural language query is tokenized and run against the index. The system ranks every chunk in the corpus using BM25 or TF-IDF and returns the top-k results. The major challenge here is the vocabulary mismatch: a user might ask about "setting up" while the code uses the term "configuration."

### Recall@k & IoU (Intersection over Union)
An evaluation metric used to grade the retrieval system. Recall measures the proportion of *actual* correct answers that your system successfully found. 
* **@k:** Means we only look at the top `k` results (e.g., top 1, 3, 5, 10).
* **IoU > 0.05:** You do not have to match exact character boundaries. As long as the retrieved snippet is from the exact same file and its character overlap with the ground truth satisfies `intersection / union > 0.05`, it is counted as a match.
* **Target Thresholds:** $\ge 80\%$ Recall@5 on docs questions, and $\ge 50\%$ Recall@5 on code questions.

### Augmentation / Prompting
The process of dynamically constructing the input for the LLM. The system takes the raw retrieved chunks, formats them nicely, appends the user's original query, and wraps them in strict system instructions.

### Generation
The final step where the LLM (`Qwen/Qwen3-0.6B`) reads the augmented prompt. Because `Qwen3-0.6B` is a small model (0.6B parameters), the answer quality relies heavily on clean context retrieval and a well-structured prompt.

### Pydantic Models
A Python library that enforces data validation at runtime using Python type hints. Ensures strict data integrity for `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`, and `RagDataset`.

### Python Fire
A Google library that turns classes and functions directly into CLI commands, converting arguments into CLI flags automatically.

### Moulinette
The automated grading binary provided by 42. It evaluates student search outputs against ground truth datasets to calculate official Recall@k scores.

---

## Currently Implemented

- **Data Models** (`src/models.py`): Complete Pydantic schemas for questions, sources, search results, and answers.
- **Chunking Strategy** (`src/indexing/chunking.py`): Markdown header splitting and Python AST/code splitting via LangChain.
- **Index Building** (`src/indexing/builder.py`): Crawls `data/raw/` and serializes `bm25_index.pkl` and `chunks.pkl` in `data/processed/`.
- **Retrieval Engine** (`src/retrieval/searcher.py`): BM25 ranking returning ranked `MinimalSource` objects.
- **LLM Generation** (`src/generation/llm.py`): `Small_llm` wrapper loading `Qwen/Qwen3-0.6B`, context formatting via `build_prompt`, and text generation via `generate_answer`.
- **Makefile** (`Makefile`): Mandatory targets `install`, `run`, `debug`, `clean`, `lint`, and `lint-strict`.
- **CLI Architecture** (`src/__main__.py` & `src/cli.py`):
  - `index`: Ingests corpus and builds BM25 index.
  - `search`: Single-query search printing top-k file locations and character ranges. (Tested & verified).
  - `search_dataset`: Batch search across dataset JSON files, producing validated `StudentSearchResults` JSON.
  - `answer`: Single-query search + LLM answer generation.
  - `answer_dataset`: Batch answer generation using `Small_llm` with `tqdm` progress tracking, producing `StudentSearchResultsAndAnswer` JSON.
- **Environment & Storage**: Large packages (`torch`, `transformers`) and caches (`uv`, Hugging Face) redirected to `/goinfre` to prevent home quota exhaustion.

---

## Current Task: Local Evaluator (`evaluate` command)

- **Objective**: Implement local Recall@k scoring based on 1D character-span IoU > 0.05.
- **Requirements**:
  - Compare student search outputs (`student_search_results_path`) against ground truth (`dataset_path`).
  - Calculate Recall@1, Recall@3, Recall@5, and Recall@10.
  - Match questions by `question_id` and evaluate source overlaps per question.
  - Display clean evaluation summary table matching moulinette format.

---

## Next Steps Planned

1. Implement IoU calculation and local evaluation function (`evaluate`).
2. Run benchmark on `dataset_docs_public.json` and `dataset_code_public.json` to verify $\ge 80\%$ (docs) and $\ge 50\%$ (code) Recall@5 thresholds.
3. Make `max_chunk_size` dynamic throughout `gather_corpus`, `chunk_markdown`, and `chunk_python`.
4. Update `README.md` to conform to all 42 subject sections.
5. Run `make lint` and ensure 100% compliance with `flake8` and `mypy`.
