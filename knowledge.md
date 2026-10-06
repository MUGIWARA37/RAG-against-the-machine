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

### Recall@k
An evaluation metric used to grade the retrieval system. Recall measures the proportion of *actual* correct answers that your system successfully found. 
* **@k:** Means we only look at the top `k` results (e.g., top 5). If the correct answer is ranked 6th, it counts as a failure for Recall@5.
* **IoU > 0.05:** In this project, you don't have to retrieve the exact character-for-character string. As long as the chunk you retrieve overlaps with the ground-truth answer by at least 5% (Intersection over Union), it counts as a success.

### Augmentation / Prompting
The process of dynamically constructing the input for the LLM. The system takes the raw retrieved chunks, formats them nicely, appends the user's original query, and wraps them in strict system instructions (e.g., "Answer this question using ONLY the context provided below. If the answer is not in the context, say 'I don't know'").

### Generation
The final step where the LLM (`Qwen/Qwen3-0.6B`) reads the augmented prompt. Because `Qwen3-0.6B` is a very small model (0.6 Billion parameters), it has limited logical reasoning capabilities. Therefore, the quality of the generation relies heavily on the quality of the retrieval and the clarity of the prompt.

### Pydantic Models
A Python library that enforces data validation at runtime using Python type hints. In a complex pipeline (Parsing -> Indexing -> Retrieval -> LLM), it is very easy for data structures to get messy. Pydantic ensures that a `MinimalSource` object always has exactly a `file_path`, a `first_character_index`, and a `last_character_index` of the correct type, failing loudly if data is malformed.

### Python Fire
A library developed by Google that automatically generates Command-Line Interfaces (CLIs) from standard Python code. It uses Python's introspection to look at your functions and turns their arguments into CLI flags (e.g., `def index(max_chunk_size=2000)` automatically becomes `--max_chunk_size`).

### Moulinette
The automated grading binary provided by the school. It runs your retrieval outputs against a hidden ground-truth dataset to strictly calculate your Recall@k scores and ensure you meet the minimum thresholds (80% for docs, 50% for code).

---

## Currently Implemented
- **Data Models**: Pydantic models for inputs and outputs (`src/models.py`).
- **Chunking Strategy**: Python and Markdown chunking with Langchain (`src/indexing/chunking.py`).
- **Index Building**: BM25 index creation and serialization (`src/indexing/builder.py`).
- **Retrieval**: Basic BM25 search matching tokenized queries against index (`src/retrieval/searcher.py`).

## Current Task: LLM Generation (`src/generation/llm.py`)
- **Objective**: Implement text generation using the `Qwen/Qwen3-0.6B` model.
- **Requirements**:
  - Receive retrieved chunks (context) and the user query.
  - Construct a strict prompt combining the context and query.
  - Run inference with `Qwen3-0.6B`.
  - Return the generated answer as a string.

## Next Steps Planned
1. Integrate Python Fire for the CLI (`cli.py`, `__main__.py`).
2. Make `max_chunk_size` dynamic via CLI arguments.
3. Wire everything up into the required CLI commands.
4. Implement a local `evaluate` function.
