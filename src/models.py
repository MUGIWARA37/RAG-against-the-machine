import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a retrieved text segment within a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents a question to be searched or answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a ground-truth question with sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset containing questions and reference answers."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents the retrieved sources for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents the generated answer and sources for a single question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represents the search results for a dataset of questions."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represents the search results and generated answers for a dataset."""

    search_results: List[MinimalAnswer]
    k: int
