from typing import List
from src.models import MinimalSource


def build_prompt(query: str, sources: List[MinimalSource], master_chunks: list[dict]) -> str:
    """
    Constructs a prompt for the LLM combining the user's query and the retrieved context.

    Args:
        query (str): The user's question.
        sources (List[MinimalSource]): The retrieved source metadata.
        master_chunks (list[dict]): The list of all chunks to extract the actual text context.

    Returns:
        str: The fully formatted prompt string ready to be fed to the model.
    
    Tips:
        - Iterate over `sources` to find the matching text in `master_chunks` (using file_path and character indices).
        - Concatenate the texts into a single context string.
        - Create a prompt template like: "Context: {context}\n\nQuestion: {query}\nAnswer:"
    """
    chunks_lookup = {
        (chunk["file_path"], chunk["first_character_index"], chunk["last_character_index"]): chunk["text"]
         for chunk in master_chunks            
        }
    
    context = []
    for source in sources:
        context.append(chunks_lookup.get((source.file_path, source.first_character_index, source.last_character_index), ""))
    
    prompt = (
        f"Context: {"\n\n".join(context)}\n\n",
        f"Qustion: {query}",
        f"Answear:"
    )
    
    return prompt
    
    
    


def generate_answer(prompt: str) -> str:
    """
    Generates an answer from the Qwen/Qwen3-0.6B model given a prompt.

    Args:
        prompt (str): The formatted prompt containing the context and the question.

    Returns:
        str: The generated natural language answer.
    
    Tips:
        - Ensure you load the tokenizer and model (e.g., `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")`).
        - Tokenize the input prompt.
        - Use the model's `generate()` method (consider setting `max_new_tokens` to limit the length).
        - Decode the generated tokens back to a string and return the newly generated part.
        - Note: To avoid reloading the model for every question, you might want to load the model
          globally or encapsulate this in a class.
    """
    # 1. Load tokenizer and model (if not already loaded).
    # 2. Tokenize the input prompt.
    # 3. Generate output using the model.
    # 4. Decode and return the output.
    pass
