from typing import Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)


def chunk_markdown(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[dict[str, Any]]:
    """
    Chunk a markdown text into smaller pieces based on headers.

    Args:
        text (str): The markdown text to be chunked.
        file_path (str): The path to the markdown file.
        max_chunk_size (int): Maximum chunk size in characters (default 2000).

    Returns:
        list[dict[str, Any]]: A list of chunked text pieces with metadata.
    """
    try:
        file_path = file_path.strip()
        if not file_path.endswith(".md"):
            raise ValueError(
                "The file path must point to a markdown file (.md)."
            )
        splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.MARKDOWN,
                    chunk_size=max_chunk_size,
                    chunk_overlap=0
        )
        chunks = splitter.split_text(text)

        extracted_data = []
        search_start = 0
        for chunk in chunks:
            start_index = text.find(chunk, search_start)
            if start_index != -1:
                end_index = start_index + len(chunk)
                extracted_data.append({
                    "file_path": file_path,
                    "first_character_index": start_index,
                    "last_character_index": end_index,
                    "text": chunk
                })
                search_start = end_index

        return extracted_data
    except Exception as e:
        raise RuntimeError(f"Error chunking markdown text: {e}")


def chunk_python(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[dict[str, Any]]:
    """
    Chunk a Python code text based on function and class definitions.

    Args:
        text (str): The Python code text to be chunked.
        file_path (str): The path to the Python file.
        max_chunk_size (int): Maximum chunk size in characters (default 2000).

    Returns:
        list[dict[str, Any]]: A list of chunked text pieces with metadata.
    """
    try:
        file_path = file_path.strip()
        if not file_path.endswith(".py"):
            raise ValueError(
                "The file path must point to a Python file (.py)."
            )

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=0
        )
        chunks = splitter.split_text(text)

        extracted_data = []
        search_start = 0
        for chunk in chunks:
            start_index = text.find(chunk, search_start)
            if start_index != -1:
                end_index = start_index + len(chunk)
                extracted_data.append({
                    "file_path": file_path,
                    "first_character_index": start_index,
                    "last_character_index": end_index,
                    "text": chunk
                })
                search_start = end_index

        return extracted_data
    except Exception as e:
        raise RuntimeError(f"Error chunking Python code text: {e}")
