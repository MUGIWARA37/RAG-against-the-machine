import uuid
from typing import List
from pydantic import BaseModel, Field



class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
    str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str