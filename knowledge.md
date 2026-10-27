# RAG Against the Machine - Knowledge Base & Defense Guide

*Author: rhlou*  
*Project: 42 Curriculum - RAG against the machine (v2.0)*

---

## 1. Core Concepts & Definitions

### 1.1 RAG (Retrieval-Augmented Generation)
Large Language Models (LLMs) are frozen in time based on their training cutoff and are prone to hallucinations when asked about private, evolving, or niche codebases. RAG solves this by turning a "closed-book exam" into an "open-book exam":
1. **Retrieve:** Queries an external corpus database to find the exact source snippets relevant to the user question.
2. **Augment:** Injects those retrieved snippets into a carefully structured prompt template.
3. **Generate:** Directs a local language model (`Qwen/Qwen3-0.6B`) to answer the question using **only** the retrieved context.

### 1.2 Corpus
The knowledge repository to be indexed. In this project, the corpus is the raw source code and documentation of `vLLM` (`data/raw/vllm-0.10.1`, ~2,867 files). A codebase corpus contains both structured logic (`.py`) and natural language prose (`.md`), requiring specialized chunking and tokenization strategies.

### 1.3 Chunking & Exact Character Offsets
LLMs have finite context windows, and retrieval requires segmenting documents into bite-sized units ($\le 2000$ characters).
* **Markdown Chunking:** Splits along header hierarchies (`#`, `##`, `###`), horizontal rules, code block delimiters, and paragraphs using `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)`.
* **Python Chunking:** Splits along grammar boundaries (top-level classes `\nclass `, functions `\ndef `, and statements) using `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`.
* **Character Offset Tracking:** Every chunk stores `file_path`, `first_character_index` (start offset), and `last_character_index` (end offset). Splitters must preserve raw newlines and whitespace without normalization so that `text.find()` accurately reproduces ground-truth character spans.

### 1.4 Inverted Index
Sequential scanning over 14,000+ chunks takes minutes per query. An **inverted index** maps each unique vocabulary token to a list of posting tuples:
$$\text{inverted\_index}[w] = [(doc\_id_1, f_1), (doc\_id_2, f_2), \dots]$$
During search, the scoring algorithm only inspects documents containing the query words, reducing retrieval time from seconds to milliseconds (< 0.02s for 100 queries).

### 1.5 Instruction-Tuned Model (Supervised Fine-Tuning - SFT)
An instruction-tuned model is a pretrained LLM that has been further fine-tuned on curated (instruction, response) pairs so it learns to interpret natural-language commands and produce the appropriate output. It's a form of supervised fine-tuning (SFT) where the loss is computed only on the response tokens, conditioning the model on the instruction.

* **ChatML (Chat Markup Language) Training:** The Qwen model family (including `Qwen/Qwen3-0.6B`) is explicitly trained on the **ChatML** format. During fine-tuning, every training sample is structured with special demarcation tokens:
  - `<|im_start|>system\n...<|im_end|>`: Defines the system role, embedding strict behavioral rules and grounding constraints.
  - `<|im_start|>user\n...<|im_end|>`: Defines the user input and reference context.
  - `<|im_start|>assistant\n...<|im_end|>`: Defines the expected model output tokens.
  
  Because Qwen's weights were conditioned on this exact markup during SFT, feeding it raw plain text causes it to fall back to generic autocomplete mode (causing hallucinations). Formatting inputs with ChatML via `tokenizer.apply_chat_template` is what allows the model to respect system-level negative constraints.

### 1.6 Small LLM Constraints & Chat Template Strategy
The mandatory model `Qwen/Qwen3-0.6B` is an ultra-small 0.6B parameter instruction-tuned model. Prompt formatting has a profound effect on its behavior:
* **The "Hallucination vs Over-Refusal" Dilemma:**
  - With naive completion prompts (`Context:... Question:... Answer:`), when an out-of-scope question is asked (e.g., "What is the capital of France?"), the base continuation mechanism bypasses the context and hallucinates facts from pre-training memory.
  - Conversely, putting rigid negative rules directly into unstructured text prompts can induce trigger bias, causing the small model to parrot refusals even on complex technical code questions.
* **The Native Solution — `tokenizer.apply_chat_template`:**
  - `Qwen/Qwen3-0.6B` comes trained with ChatML role markup (`<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`).
  - By using `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`, we cleanly separate:
    1. **System Role:** Explicit grounding instructions: *"You are a code and documentation assistant. Answer the question using only the context provided. If the answer is not mentioned in the context, say that the context does not contain enough information to answer."*
    2. **User Role:** Retrieved context chunks followed by the target question.
    3. **Assistant Role:** Generation prompt initiated with empty thinking tags `<think>\n\n</think>`, avoiding runaway reasoning loops on CPU.
  - This achieves both goals simultaneously: out-of-scope queries produce explicit refusals, while technical queries (such as LoRA endpoints, MoE activation formats, and PagedAttention mechanisms) are answered with high fidelity.
* **Separation of Concerns:** Unanswerable questions with zero retrieval hits are handled defensively in application code (`if not item.retrieved_sources`), returning the standard refusal message instantly without relying on the LLM to self-refuse.
* **Repetition Stripping:** The generation wrapper strips subsequent prompt delimiters (`\nAnswer:`, `\nQuestion:`, `</think>`) to deliver crisp, single-answer outputs.

#### Code Example: How to Use `apply_chat_template`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# 2. Build structured role messages
messages = [
    {
        "role": "system",
        "content": (
            "You are a code and documentation assistant. "
            "Answer the question using only the context provided. "
            "If the answer is not mentioned in the context, say that the "
            "context does not contain enough information to answer."
        ),
    },
    {
        "role": "user",
        "content": f"Context:\n{joined_context}\n\nQuestion: {query}",
    },
]

# 3. Apply the ChatML template
prompt_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,             # Returns formatted string
    add_generation_prompt=True, # Appends '<|im_start|>assistant\n'
    enable_thinking=False,      # Closes <think> tags for fast direct answer
)

# 4. Generate answer
inputs = tokenizer(prompt_str, return_tensors="pt")
prompt_len = inputs.input_ids.shape[1]

with torch.inference_mode():
    output_tokens = model.generate(**inputs, max_new_tokens=100)

answer = tokenizer.decode(
    output_tokens[0][prompt_len:], skip_special_tokens=True
).strip()
```

#### Rendered ChatML Prompt Structure

The template transforms the message objects into the exact text sequence:

```text
<|im_start|>system
You are a code and documentation assistant. Answer the question using only the context provided. If the answer is not mentioned in the context, say that the context does not contain enough information to answer.<|im_end|>
<|im_start|>user
Context:
... [retrieved code / doc chunks] ...

Question: What is PagedAttention?<|im_end|>
<|im_start|>assistant
<think>

</think>
```

---

## 2. Mathematical Formulations

### 2.1 Comparison: TF vs. TF-IDF vs. BM25

| Method | Formula | Flaws & Limitations |
|---|---|---|
| **TF**<br/>*(Term Frequency)* | $\text{TF}(t, D) = f(t, D)$ | Treats all words equally. High-frequency stop words (e.g. `"to"`, `"a"`, `"def"`) dominate search results over rare, high-signal terms. |
| **TF-IDF**<br/>*(Term Frequency - Inverse Document Frequency)* | $\text{TF-IDF}(t, D) = f(t, D) \cdot \log\left(\frac{N}{n(t)}\right)$ | **1. No Term Saturation:** Linear scaling allows keyword-stuffed documents to artificially dominate.<br/>**2. Document Length Bias:** Unfairly favors long documents simply because they contain more total words. |
| **BM25**<br/>*(Best Matching 25)* | $\sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{\|D\|}{\text{avgdl}}\right)}$ | Resolves both flaws: $k_1$ enforces an asymptotic saturation ceiling on repeated words, and $b$ normalizes against average document length ($\text{avgdl}$). |

#### Feature Comparison Summary:

| Feature | Term Frequency (TF) | TF-IDF | BM25 |
|---|:---:|:---:|:---:|
| Counts word occurrences | ✅ Yes | ✅ Yes | ✅ Yes |
| Penalizes common words (via IDF) | ❌ No | ✅ Yes | ✅ Yes |
| Caps repeated words (Saturation via $k_1$) | ❌ No | ❌ No | ✅ Yes |
| Normalizes document length (via $b$ and $\text{avgdl}$) | ❌ No | ❌ No | ✅ Yes |

### 2.2 Robertson-Spärck Jones BM25 Algorithm
BM25 (Best Matching 25) balances term frequency saturation with document length normalization:

$$\text{Score}(D, Q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$: Frequency of query term $q_i$ in chunk $D$.
* $|D|$: Length (token count) of chunk $D$.
* $\text{avgdl}$: Average token length across all chunks in the corpus.
* $k_1 = 1.5$: Term frequency saturation parameter. Controls how quickly the score plateau is reached.
* $b = 0.75$: Length normalization parameter. $b=1$ fully normalizes for document length; $b=0$ ignores document length.

### 2.3 Robertson-Spärck Jones Inverse Document Frequency (RSJ-IDF)
To prevent negative IDF values for words that appear in over 50% of documents, we use the smoothed RSJ-IDF formula:

$$\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$$

Where:
* $N$: Total number of chunks in the corpus.
* $n(q_i)$: Number of chunks containing term $q_i$.

### 2.4 1D Character-Span IoU (Intersection over Union)
Measures the overlap between retrieved character bounds $[s_1, e_1]$ and ground truth bounds $[s_2, e_2]$ within the same file:

$$\text{Intersection} = \max(0, \min(e_1, e_2) - \max(s_1, s_2))$$
$$\text{Union} = (e_1 - s_1) + (e_2 - s_2) - \text{Intersection}$$
$$\text{IoU} = \frac{\text{Intersection}}{\text{Union}}$$

A retrieved source is a **match** if and only if:
1. `retrieved.file_path == ground_truth.file_path`
2. $\text{IoU} > 0.05$ (strictly greater than 5% overlap)

### 2.5 Recall@k
$$\text{Recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{matched ground truth sources in top } k|}{|\text{total ground truth sources for } q|}$$

---

## 3. Code-Aware Tokenizer Strategy

Standard natural language tokenizers fail on codebases because:
1. Queries use spaces: `"fused batched moe"`
2. Code identifiers use conventions: `fused_batched_moe` or `FusedBatchedMoe`

### How Our Tokenizer Bridges the Gap:
1. **CamelCase Splitting:** Replaces regex `([a-z0-9])([A-Z])` with `\1 \2` (`AsyncLLM` $\rightarrow$ `Async LLM`).
2. **Word Extraction:** Matches alphanumeric tokens `[a-zA-Z0-9_]+` in lowercase.
3. **Snake_Case Decomposition:** For every word containing underscores, indexes both the composite identifier (`fused_batched_moe`) and all individual sub-words (`fused`, `batched`, `moe`).
4. **File Path Ingestion:** Prepends `file_path` to chunk text before tokenizing (`"vllm/attention/ops/flash.py ..."`), allowing file references in queries to score highly.

---

## 4. Final Benchmark Results

All metrics exceed the 42 subject requirements:

| Evaluation Metric | Subject Requirement | Achieved Result | Evaluation Status |
|---|---|---|---|
| **Documentation Recall@5** | $\ge 80.0\%$ | **80.0%** | ✅ **PASS** |
| **Code Recall@5** | $\ge 50.0\%$ | **75.8%** | ✅ **PASS** |
| **Recall@1 (Code / Docs)** | - | 41.4% / 62.0% | - |
| **Recall@3 (Code / Docs)** | - | 67.7% / 75.0% | - |
| **Recall@10 (Code / Docs)** | - | 83.8% / 82.0% | - |
| **Corpus Indexing Time** | $\le 5\text{ minutes}$ | **~8 seconds** | ✅ **PASS** |
| **Search Throughput (100 Qs)** | $\le 90\text{ seconds}$ | **< 0.02 seconds** | ✅ **PASS** |

---

## 5. Defensive Edge Case Handling

The project handles edge cases gracefully without throwing raw Python tracebacks:

1. **`k <= 0`:** Warns user and returns an empty result set cleanly.
2. **Empty or Whitespace-only Queries:** Warns user and exits without crashing.
3. **Missing Index Files (`chunks.pkl`, `bm25_index.pkl`):** Catches `FileNotFoundError`, prints `Please run 'index' first.`, and exits cleanly.
4. **File Permissions (`chmod 000`):** Catches `PermissionError` when reading datasets, reading index files, or writing output files, outputting a clean error message.
5. **Corrupted or Malformed JSON:** Catches `json.JSONDecodeError` with descriptive message.
6. **Corpus File Permission Errors:** Skips unreadable files with a logged warning during ingestion without aborting the build.

---

## 6. Codebase Architecture & Norm Compliance

### 6.1 Strict Typing & Norm Checks
* `make lint`: Runs `flake8 src/` and `mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`. **Result: 0 errors.**
* `make lint-strict`: Runs `flake8 src/` and `mypy src/ --strict --ignore-missing-imports`. **Result: 0 errors.**
* Every line $\le 79$ characters (PEP 8 standard).
* Every class, method, and function has Google/NumPy style PEP 257 docstrings documenting `Args:` and `Returns:`.

### 6.2 Module Structure
```text
src/
├── __init__.py
├── __main__.py               # CLI entrypoint with fire.Fire(Cli)
├── cli.py                    # 6 Fire subcommands with defensive checks
├── models.py                 # Pydantic V2 data models
├── indexing/
│   ├── __init__.py
│   ├── bm25.py               # Custom BM25 ranking algorithm from scratch
│   ├── builder.py            # Corpus crawler & index serializer
│   ├── chunking.py           # Language-specific Markdown & Python splitters
│   └── tokenizer.py          # Code-aware sub-token tokenizer
├── retrieval/
│   ├── __init__.py
│   ├── evaluator.py          # 1D IoU computation and Recall@k evaluator
│   └── searcher.py           # Fast inverted index query engine
└── generation/
    ├── __init__.py
    └── llm.py                # Local Qwen/Qwen3-0.6B wrapper & prompter
```

---

## 7. Defense Questions & Preparation

### Q1: Why did you implement BM25 from scratch instead of using `rank-bm25`?
> *Answer:* `rank-bm25` is an external blackbox package that does not compute inverted indices efficiently across large corpora and does not expose internal term scoring easily. Writing BM25 in pure Python using an inverted index (`defaultdict(list)`) allows us to achieve sub-millisecond retrieval (< 0.02s for 100 questions), guarantees no external dependencies, and demonstrates full understanding of RSJ-IDF, term saturation ($k_1$), and length normalization ($b$).

### Q2: Why was your code recall initially low (15%), and how did you fix it?
> *Answer:* Code questions use natural language with spaces (e.g., "fused batched moe"), whereas Python code uses `snake_case` (e.g., `fused_batched_moe`). A naive tokenizer treats `fused_batched_moe` as a single token that never matches "fused" or "batched". By decomposing `snake_case` into sub-tokens and indexing the `file_path`, code recall jumped from 15% to 75.8%.

### Q3: Why did documentation recall drop to 1% when using `MarkdownHeaderTextSplitter`?
> *Answer:* `MarkdownHeaderTextSplitter` modified newline characters (converting `\n\n` into `  \n`). When mapping chunks back to the original file to obtain `first_character_index` and `last_character_index`, `text.find()` failed, dropping chunks or producing invalid character offsets. Switching to `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` preserved exact text byte sequences and boosted docs recall to 80.0%.

### Q4: How does 1D IoU work and why is the threshold 0.05?
> *Answer:* 1D IoU computes the ratio of character overlap between the retrieved chunk $[s_1, e_1]$ and the ground truth snippet $[s_2, e_2]$:
> $$\text{IoU} = \frac{\max(0, \min(e_1, e_2) - \max(s_1, s_2))}{(e_1 - s_1) + (e_2 - s_2) - \text{Intersection}}$$
> The 0.05 threshold means that if at least 5% of the characters overlap and the `file_path` matches, the retrieved snippet contains the relevant context and counts as a successful hit.

### Q5: How did you solve the disk space limit on 42 school computers?
> *Answer:* 42 user home directories have strict 5GB quota limits. PyTorch and Hugging Face weights exceed this limit. We automated cache routing in `Makefile` (`make install`): when `~/goinfre/` is detected, it dynamically creates `~/goinfre/` cache directories (`UV_CACHE_DIR`, `HF_HOME`, and `rag-venv`), symlinking `.venv` so the entire 5.5GB environment runs off local high-speed disk without consuming the user's home quota. On non-42 machines, it gracefully falls back to local storage.

### Q6: How do you handle unanswerable questions or model hallucinations with small models?
> *Answer:* Ultra-small models like `Qwen/Qwen3-0.6B` have limited reasoning capacity and tend to autocomplete from pre-training memory if prompts lack role boundaries, or conversely suffer from over-refusal if negative instructions are placed in unstructured text. We resolve this with a two-tier approach:
> 1. **Application-level defense:** If retrieval returns no sources (`not item.retrieved_sources`), the CLI immediately outputs the standard refusal without invoking the LLM, saving CPU cycles.
> 2. **Native ChatML template (`tokenizer.apply_chat_template`):** When sources are present, we wrap the prompt into formal `system` and `user` roles using the tokenizer's chat template. The system prompt instructs the model to refuse if the information is absent from the context. This stops hallucinations on out-of-scope questions while preserving high-accuracy answering on technical code questions.
