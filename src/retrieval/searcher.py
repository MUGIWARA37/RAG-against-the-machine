import pickle
from pathlib import Path

from src.indexing.tokenizer import tokenizer
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)


def search_dataset(
    questions: list[UnansweredQuestion],
    k: int = 5,
    folder_path: Path = Path("data/processed"),
) -> StudentSearchResults:
    """
    Search the dataset for the given questions and return the top k results.

    Args:
        questions (list[UnansweredQuestion]): Questions to search for.
        k (int): Number of top results to return for each question.
        folder_path (Path): Path to the processed index directory.

    Returns:
        StudentSearchResults: The search results for the given questions.
    """
    if k <= 0:
        print("Warning: 'k' must be greater than 0.")
        return StudentSearchResults(search_results=[], k=k)

    try:
        open_index_path = folder_path / "bm25_index.pkl"
        open_chunks_path = folder_path / "chunks.pkl"
        with open(open_index_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(open_chunks_path, "rb") as f:
            master_chunks = pickle.load(f)
    except FileNotFoundError:
        print(
            f"Error: Index files not found in '{folder_path}'. "
            "Please run 'index' first."
        )
        return StudentSearchResults(search_results=[], k=k)
    except PermissionError:
        print(f"Error: Permission denied accessing '{folder_path}'.")
        return StudentSearchResults(search_results=[], k=k)

    results = []
    for question in questions:
        tokenized_query = tokenizer(question.question)

        top_chunks = bm25.get_top_n(tokenized_query, master_chunks, n=k)

        scored_results = []
        for chunk in top_chunks:
            source = MinimalSource(
                first_character_index=chunk["first_character_index"],
                file_path=chunk["file_path"],
                last_character_index=chunk["last_character_index"],
            )
            scored_results.append(source)

        result = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=scored_results,
        )
        results.append(result)

    return StudentSearchResults(search_results=results, k=k)
