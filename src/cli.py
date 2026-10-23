import json
import pickle
from pathlib import Path
from tqdm import tqdm

from src.indexing.builder import gather_corpus, build_and_save_index
from src.retrieval.searcher import search_dataset as run_search
from src.retrieval.evaluator import evaluate_search_results
from src.generation.llm import Small_llm, build_prompt, generate_answer
from src.models import (
    UnansweredQuestion,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)


class Cli:
    def index(self, max_chunk_size=2000):
        print(f"Starting indexing with max_chunk_size={max_chunk_size}...")
        chunks = gather_corpus("data/raw")
        build_and_save_index(chunks, save_dir="data/processed")
        print("Ingestion complete! Index saved under data/processed/")

    def search(self, query, k=5):
        question = UnansweredQuestion(question=query)
        results = run_search([question], k)
        for source in results.search_results[0].retrieved_sources:
             print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")


    def search_dataset(self, dataset_path, k=5, save_directory="data/output/search_results"):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        raw_questions = dataset.get("rag_questions", [])

        questions = [
            UnansweredQuestion(
                question_id=item["question_id"],
                question=item["question"]
            )
            for item in tqdm(raw_questions)
        ]

        results =  run_search(questions, k)

        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(dataset_path).name
        output_path = output_dir / file_name

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=2))
        
        print(f"Saved student_search_results to {output_path}")


    def answer(self, query, k=5):
        """Answer a single query using retrieved context and Qwen."""
        question = UnansweredQuestion(question=query)
        results = run_search([question], k=k)
        retrieved_sources = results.search_results[0].retrieved_sources
        
        with open("data/processed/chunks.pkl", "rb") as f:
            master_chunks = pickle.load(f)

        prompt = build_prompt(query, retrieved_sources, master_chunks)

        model = Small_llm()
        ans = generate_answer(prompt, model)

        print(f"\nQuestion: {query}\n")
        print(f"Answer:\n{ans}\n")

    def answer_dataset(self, student_search_results_path, save_directory="data/output/search_results_and_answer"):
        with open(student_search_results_path, "r") as f:
            data = json.load(f)
        search_results = StudentSearchResults.model_validate(data)

        with open("data/processed/chunks.pkl", "rb") as f:
            master_chunks = pickle.load(f)

        model = Small_llm()
        answers = []

        for item in tqdm(search_results.search_results, desc="Generating answers", unit="answer"):
            prompt = build_prompt(item.question, item.retrieved_sources, master_chunks)
            ans = generate_answer(prompt, model)
            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=ans
                )
            )

        final_result = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k
        )

        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(student_search_results_path).name
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_result.model_dump_json(indent=2))
        
        print(f"Saved student_search_results_and_answer to {output_path}")


    
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        """Report recall@k against a ground-truth dataset."""
        evaluate_search_results(student_search_results_path, dataset_path)