import json
from pathlib import Path
from typing import Any, Dict, Optional


class QueryCache:
    """
    Persistent on-disk cache for query retrieval results and generated answers.
    """

    def __init__(
        self, cache_file: str = "data/processed/query_cache.json"
    ) -> None:
        """
        Initializes the cache manager.

        Args:
            cache_file (str): Path to the cache storage JSON file.
        """
        self.cache_path = Path(cache_file)
        self.cache: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Loads cached queries from disk if the file exists."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.cache = {}

    def _save(self) -> None:
        """Saves current cache dictionary to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except OSError:
            pass

    def get(self, prefix: str, query: str, k: int) -> Optional[Any]:
        """
        Retrieves a cached value for the given query and k.

        Args:
            prefix (str): Cache namespace (e.g. 'search' or 'answer').
            query (str): The search query.
            k (int): Number of sources requested.

        Returns:
            Optional[Any]: Cached object or None if missing.
        """
        key = f"{prefix}:{query.strip().lower()}:{k}"
        return self.cache.get(key)

    def set(self, prefix: str, query: str, k: int, value: Any) -> None:
        """
        Stores a value in the cache and persists to disk.

        Args:
            prefix (str): Cache namespace (e.g. 'search' or 'answer').
            query (str): The search query.
            k (int): Number of sources requested.
            value (Any): The JSON-serializable value to cache.
        """
        key = f"{prefix}:{query.strip().lower()}:{k}"
        self.cache[key] = value
        self._save()

    def clear(self) -> None:
        """Clears all cached entries and deletes the cache file."""
        self.cache = {}
        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
            except OSError:
                pass
