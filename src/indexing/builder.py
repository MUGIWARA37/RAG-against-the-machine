import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.indexing.chunking import chunk_markdown, chunk_python


def gather_corpus(directory_path: str) -> list[dict[str, Any]]:
    """
    Crawls a directory, reads .md and .py files, and chunks them.
    """
    master_chunks: list[dict[str, Any]] = []

    base_path = Path(directory_path)
    all_files = [p for p in base_path.rglob("*") if p.is_file()]

    for file_path in tqdm(all_files, desc="Chunking files"):
        ext = file_path.suffix

        if ext in [".md", ".py"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                if ext == ".md":
                    chunks = chunk_markdown(text, str(file_path))
                else:
                    chunks = chunk_python(text, str(file_path))
                master_chunks.extend(chunks)
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    return master_chunks


def build_and_save_index(
    master_chunks: list[dict[str, Any]],
    save_dir: str = "data/processed"
) -> None:
    """
    Tokenizes the corpus, builds a BM25 index, and saves it to disk.
    """
    print("Tokenizing corpus and building BM25 index...")
    tokenized_corpus = [
        chunk["text"].lower().split() for chunk in master_chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"Saving index to {save_dir}...")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    index_file = save_path / "bm25_index.pkl"
    with open(index_file, "wb") as f:
        pickle.dump(bm25, f)

    master_index = save_path / "chunks.pkl"
    with open(master_index, "wb") as f:
        pickle.dump(master_chunks, f)

    print("Indexing complete!")
