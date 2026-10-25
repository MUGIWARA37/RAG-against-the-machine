import math
from collections import Counter, defaultdict
from typing import Any, List, Tuple, Dict

class Bm25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_size = len(corpus)
        self.k1 = k1
        self.b = b

        self.doc_lens = [len(doc) for doc in corpus]
        total_tokens = sum(self.doc_lens)
        self.avgdl = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0


        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_freqs: Dict[str, int] = defaultdict(int)

        for doc_id, doc in enumerate(corpus):
            freqs = Counter(doc)
            for word, freq in freqs.items():
                self.inverted_index[word].append((doc_id, freq))
                self.doc_freqs[word] += 1

        self.idf: Dict[str, float] = {}
        for word, n_w in self.doc_freqs.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - n_w + 0.5)/ (n_w + 0.5))


    def get_top_n(self, query: List[str],  documents: List[Any], n: int = 5) -> List[Any]:

        scores: Dict[int, float] = defaultdict(float)

        for word in query:
            if word not in self.inverted_index:
                continue

            idf = self.idf[word]

            for doc_id, freq in self.inverted_index[word]:
                doc_len  = self.doc_lens[doc_id]
                numerator = freq * (self.k1 + 1.0)
                denomirator = freq + self.k1 * (1.0 -self.b + self.b *(doc_len) / self.avgdl)

                scores[doc_id] += idf * (numerator/denomirator)

        top_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)[:n]

        return [documents[doc_id] for doc_id in top_doc_ids]