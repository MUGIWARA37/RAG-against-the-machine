import re
from typing import List


def tokenizer(text: str) -> List[str]:
    """
    Tokenizes text with code awareness for camelCase and snake_case terms.

    Args:
        text (str): Input text or code snippet to tokenize.

    Returns:
        List[str]: List of extracted lowercase tokens and sub-tokens.
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)

    words = re.findall(r"[a-zA-Z0-9_]+", spaced.lower())

    tokens: List[str] = []
    for word in words:
        tokens.append(word)
        if "_" in word:
            for part in word.split("_"):
                if len(part) > 1:
                    tokens.append(part)

    return tokens
