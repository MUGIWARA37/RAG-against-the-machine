import os
from typing import Any, List
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

        home_goinfre = os.path.expanduser("~/goinfre")
        if "HF_HOME" not in os.environ and os.path.isdir(home_goinfre):
            goinfre_hf = os.path.join(home_goinfre, ".cache", "huggingface")
            os.makedirs(goinfre_hf, exist_ok=True)
            os.environ["HF_HOME"] = goinfre_hf

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.to(self.device)  # type: ignore[arg-type]
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
        tokenized_prompt = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)

        tokenized_prompt_lengh = tokenized_prompt.input_ids.shape[1]

        with torch.inference_mode():
            logits = self.model.generate(  # type: ignore[misc]
                **tokenized_prompt,
                max_new_tokens=100
            )[0][tokenized_prompt_lengh:]

        output = self.tokenizer.decode(logits, skip_special_tokens=True)
        text = str(output).strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        for delimiter in (
            "\nAnswer:",
            "\nQuestion:",
            "\nThe answer is:",
            "\nBased on the answer",
        ):
            if delimiter in text:
                text = text.split(delimiter)[0].strip()

        return text


def build_prompt(
    query: str,
    sources: List[MinimalSource],
    master_chunks: list[dict[str, Any]],
    chunks_lookup: dict[tuple[str, int, int], str] | None = None,
    tokenizer: Any | None = None,
) -> str:
    """
    Constructs a prompt for the LLM combining the user's query and the
    retrieved context using a chat template.

    Args:
        query (str): The user's question.
        sources (List[MinimalSource]): The retrieved source metadata.
        master_chunks (list[dict[str, Any]]): All chunk dictionaries.
        chunks_lookup (dict[tuple[str, int, int], str] | None): Optional
            precomputed lookup mapping from (path, start, end) to text.
        tokenizer (Any | None): Optional tokenizer instance implementing
            apply_chat_template.

    Returns:
        str: The fully formatted prompt string ready to be fed to the model.
    """
    if chunks_lookup is None:
        chunks_lookup = {
            (
                chunk["file_path"],
                chunk["first_character_index"],
                chunk["last_character_index"],
            ): str(chunk["text"])
            for chunk in master_chunks
        }

    context = []
    for source in sources:
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index,
        )
        context.append(chunks_lookup.get(key, ""))

    joined_context = "\n\n".join(context)

    system_prompt = (
        "You are a code and documentation assistant. "
        "Answer the question using only the context provided. "
        "If the answer is not mentioned in the context, say that the "
        "context does not contain enough information to answer."
    )
    user_content = f"Context:\n{joined_context}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


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
