import json
from typing import Any, Dict, List

from src.models import MinimalSource, StudentSearchResults


def compute_iou(
    start1: int, end1: int, start2: int, end2: int
) -> float:
    """
    Compute 1D Intersection over Union (IoU) between two character spans.

    Args:
        start1 (int): Start index of the first character span.
        end1 (int): End index of the first character span.
        start2 (int): Start index of the second character span.
        end2 (int): End index of the second character span.

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    intersection = max(0, min(end1, end2)) - max(0, max(start1, start2))
    if intersection <= 0:
        return 0.0

    l1 = max(0, end1 - start1)
    l2 = max(0, end2 - start2)

    union = l1 + l2 - intersection

    return intersection / union if union > 0 else 0.0


def is_source_match(
    retrieved: MinimalSource,
    ground_truth: Dict[str, Any],
    iou_threshold: float = 0.05
) -> bool:
    """
    Check if a retrieved source matches a ground truth source.

    A match requires identical file_path and an IoU strictly greater than
    the specified threshold (default 0.05).

    Args:
        retrieved (MinimalSource): Retrieved source from student search.
        ground_truth (Dict[str, Any]): Reference source dictionary.
        iou_threshold (float): Minimum IoU required for a match.

    Returns:
        bool: True if source matches, False otherwise.
    """
    if retrieved.file_path != ground_truth.get("file_path"):
        return False

    iou = compute_iou(
        retrieved.first_character_index,
        retrieved.last_character_index,
        ground_truth.get("first_character_index"),
        ground_truth.get("last_character_index")
    )

    return iou > iou_threshold


def evaluate_search_results(
    student_search_results_path: str,
    dataset_path: str,
    k_values: List[int] = [1, 3, 5, 10],
    iou_threshold: float = 0.05
) -> Dict[int, float]:
    """
    Evaluate student search results against ground truth dataset.

    Args:
        student_search_results_path (str): Path to student search results JSON.
        dataset_path (str): Path to ground-truth AnsweredQuestions JSON.
        k_values (List[int]): List of k cutoffs to evaluate recall for.
        iou_threshold (float): Minimum IoU overlap (default 0.05).

    Returns:
        Dict[int, float]: Mapping of k to average Recall@k score.
    """
    with open(student_search_results_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)
    student_obj = StudentSearchResults.model_validate(student_data)

    with open(dataset_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_by_id = {
        item["question_id"]: item.get("sources", [])
        for item in gt_data.get("rag_questions", [])
    }

    recall_sums = {k: 0.0 for k in k_values}
    evaluated_count = 0

    for result in student_obj.search_results:
        gt_sources = gt_by_id.get(result.question_id, [])
        if not gt_sources:
            continue

        evaluated_count += 1
        retrieved = result.retrieved_sources

        for k in k_values:
            top_k = retrieved[:k]

            found_count = 0
            for gt_src in gt_sources:
                if any(is_source_match(ret_src, gt_src, iou_threshold) for ret_src in top_k):
                    found_count += 1

            recall_sums[k] += found_count / len(gt_sources)

    recalls = {}
    for k in k_values:
        recalls[k] = recall_sums[k] / evaluated_count if evaluated_count > 0 else 0.0

    print("Student data is valid: True")
    print(f"Total number of questions evaluated: {evaluated_count}")
    print("\nEvaluation Results\n" + "=" * 40)
    for k in k_values:
        score = recalls[k]
        print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")

    return recalls