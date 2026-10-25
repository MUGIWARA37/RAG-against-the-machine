import json
import pickle
from pathlib import Path
from tqdm import tqdm

from src.generation.llm import Small_llm, build_prompt, generate_answer
from src.indexing.builder import build_and_save_index, gather_corpus
from src.models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retrieval.evaluator import evaluate_search_results
from src.retrieval.searcher import search_dataset as run_search


class Cli:
    """Command-line interface for the RAG against the machine system."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingests and indexes the raw corpus.

        Args:
            max_chunk_size (int): Maximum character size per chunk.
        """
        print(f"Starting indexing with max_chunk_size={max_chunk_size}...")
        chunks = gather_corpus("data/raw", max_chunk_size)
        build_and_save_index(chunks, save_dir="data/processed")
        print("Ingestion complete! Index saved under data/processed/")

    def search(self, query: str, k: int = 5) -> None:
        """
        Searches the indexed corpus for a single natural language query.

        Args:
            query (str): The query string to search for.
            k (int): Number of top matching sources to display.
        """
        if k <= 0:
            print("Error: 'k' must be greater than 0.")
            return

        if not str(query).strip():
            print("Error: Query cannot be empty.")
            return

        question = UnansweredQuestion(question=query)
        results = run_search([question], k)

        if (
            not results.search_results
            or not results.search_results[0].retrieved_sources
        ):
            print("No matching sources found for the given query.")
            return

        for source in results.search_results[0].retrieved_sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """
        Searches for all questions in a dataset and saves structured results.

        Args:
            dataset_path (str): Path to input dataset JSON file.
            k (int): Number of top matching sources to retrieve.
            save_directory (str): Directory where search results are saved.
        """
        if k <= 0:
            print("Error: 'k' must be greater than 0.")
            return

        if not Path(dataset_path).is_file():
            print(f"Error: Dataset file not found at '{dataset_path}'.")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except PermissionError:
            print(
                "Error: Permission denied when attempting to read "
                f"'{dataset_path}'."
            )
            return
        except json.JSONDecodeError:
            print(
                f"Error: Failed to parse '{dataset_path}'. "
                "File contains invalid JSON."
            )
            return
        except Exception as e:
            print(f"Error reading '{dataset_path}': {e}")
            return

        raw_questions = dataset.get("rag_questions", [])

        questions = [
            UnansweredQuestion(
                question_id=item["question_id"],
                question=item["question"],
            )
            for item in tqdm(raw_questions)
        ]

        results = run_search(questions, k)

        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(dataset_path).name
        output_path = output_dir / file_name

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=2))
        except PermissionError:
            print(f"Error: Permission denied writing to '{output_path}'.")
            return

        print(f"Saved student_search_results to {output_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """
        Answers a single query using retrieved context and local LLM.

        Args:
            query (str): The query string to answer.
            k (int): Number of context sources to retrieve.
        """
        if k <= 0:
            print("Error: 'k' must be greater than 0.")
            return

        if not str(query).strip():
            print("Error: Query cannot be empty.")
            return

        chunks_path = Path("data/processed/chunks.pkl")
        if not chunks_path.is_file():
            print(
                "Error: Index files not found in 'data/processed'. "
                "Please run 'index' first."
            )
            return

        question = UnansweredQuestion(question=query)
        results = run_search([question], k=k)

        if (
            not results.search_results
            or not results.search_results[0].retrieved_sources
        ):
            print("No matching context found to answer the query.")
            return

        retrieved_sources = results.search_results[0].retrieved_sources

        try:
            with open("data/processed/chunks.pkl", "rb") as f:
                master_chunks = pickle.load(f)
        except PermissionError:
            print(
                "Error: Permission denied reading "
                "'data/processed/chunks.pkl'."
            )
            return

        prompt = build_prompt(query, retrieved_sources, master_chunks)

        model = Small_llm()
        ans = generate_answer(prompt, model)

        print(f"\nQuestion: {query}\n")
        print(f"Answer:\n{ans}\n")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """
        Generates grounded answers for all questions in search results.

        Args:
            student_search_results_path (str): Path to search results JSON.
            save_directory (str): Directory where answers are saved.
        """
        if not Path(student_search_results_path).is_file():
            print(
                "Error: Search results file not found at "
                f"'{student_search_results_path}'."
            )
            return

        chunks_path = Path("data/processed/chunks.pkl")
        if not chunks_path.is_file():
            print(
                "Error: Index chunks not found in 'data/processed'. "
                "Please run 'index' first."
            )
            return

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            search_results = StudentSearchResults.model_validate(data)
        except PermissionError:
            print(
                "Error: Permission denied when attempting to read "
                f"'{student_search_results_path}'."
            )
            return
        except json.JSONDecodeError:
            print(
                f"Error: '{student_search_results_path}' contains "
                "invalid JSON."
            )
            return
        except Exception as e:
            print(f"Error loading search results: {e}")
            return

        try:
            with open("data/processed/chunks.pkl", "rb") as f:
                master_chunks = pickle.load(f)
        except PermissionError:
            print(
                "Error: Permission denied reading "
                "'data/processed/chunks.pkl'."
            )
            return

        chunks_lookup = {
            (
                c["file_path"],
                c["first_character_index"],
                c["last_character_index"],
            ): str(c["text"])
            for c in master_chunks
        }

        model = Small_llm()
        answers = []

        for item in tqdm(
            search_results.search_results,
            desc="Generating answers",
            unit="answer",
        ):
            prompt = build_prompt(
                item.question,
                item.retrieved_sources,
                master_chunks,
                chunks_lookup,
            )
            ans = generate_answer(prompt, model)
            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=ans,
                )
            )

        final_result = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )

        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(student_search_results_path).name

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_result.model_dump_json(indent=2))
        except PermissionError:
            print(f"Error: Permission denied writing to '{output_path}'.")
            return

        print(
            "Saved student_search_results_and_answer to "
            f"{output_path}"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """
        Evaluates Recall@k of search results against a ground-truth dataset.

        Args:
            student_search_results_path (str): Path to search results JSON.
            dataset_path (str): Path to reference AnsweredQuestions JSON.
        """
        evaluate_search_results(student_search_results_path, dataset_path)
