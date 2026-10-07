from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models import MinimalSource


class Small_llm:
    """
    A wrapper class to manage the loading and text generation of a causal
    language model using Hugging Face Transformers.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Initializes the tokenizer and model.

        Args:
            model_id (str): The Hugging Face model identifier.
                Defaults to "Qwen/Qwen3-0.6B".

        Raises:
            Exception: If the model cannot be loaded from Hugging Face.
        """
        self.model_id = model_id

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32
            )
        except Exception:
            raise Exception(
                f"Error: loading the model {self.model_id} "
                "make sure this model exists !!"
            )

    def generate(self, prompt: str) -> str:
        """
        Generates a natural language response based on the provided prompt.

        Args:
            prompt (str): The formatted prompt string.

        Returns:
            str: The generated text, with the original prompt stripped out.
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")

        tokenized_prompt_lengh = tokenized_prompt.input_ids.shape[1]

        logits = self.model.generate(
            **tokenized_prompt,
            max_new_tokens=200
        )[0][tokenized_prompt_lengh:]

        output = self.tokenizer.decode(logits, skip_special_tokens=True)

        return str(output).strip()


def build_prompt(
    query: str, sources: List[MinimalSource], master_chunks: list[dict]
) -> str:
    """
    Constructs a prompt for the LLM combining the user's query and the
    retrieved context.

    Args:
        query (str): The user's question.
        sources (List[MinimalSource]): The retrieved source metadata.
        master_chunks (list[dict]): The list of all chunks to extract text.

    Returns:
        str: The fully formatted prompt string ready to be fed to the model.
    """
    chunks_lookup = {
        (
            chunk["file_path"],
            chunk["first_character_index"],
            chunk["last_character_index"]
        ): chunk["text"]
        for chunk in master_chunks
    }

    context = []
    for source in sources:
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index
        )
        context.append(chunks_lookup.get(key, ""))

    joined_context = "\n\n".join(context)

    prompt = f"Context: {joined_context}\n\nQuestion: {query}\nAnswer:"

    return prompt


def generate_answer(prompt: str, model: Small_llm) -> str:
    """
    Generates an answer from the language model given a formatted prompt.

    Args:
        prompt (str): The formatted prompt containing context and question.
        model (Small_llm): The instantiated model wrapper class.

    Returns:
        str: The generated natural language answer.
    """
    return model.generate(prompt)
