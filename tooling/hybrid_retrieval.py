"""知识库切片的 BM25 召回和按名次融合。"""
import math
import re
import unicodedata
from collections import Counter, defaultdict


_HAN = r"[\u3400-\u9fff]"
_TOKENS = re.compile(rf"{_HAN}+|[a-z0-9]+(?:[._-][a-z0-9]+)*")
_BM25_K1 = 1.2
_BM25_B = 0.75
_RRF_K = 60


def tokenize(text: str) -> list[str]:
    tokens = []
    for part in _TOKENS.findall(unicodedata.normalize("NFKC", text).casefold()):
        if len(part) > 1 and re.match(_HAN, part):
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self._chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        self._postings = defaultdict(dict)
        lengths = {}
        for chunk_id, chunk in self._chunks.items():
            terms = Counter(tokenize(f"{chunk['title']}\n{chunk['content']}"))
            lengths[chunk_id] = sum(terms.values())
            for term, frequency in terms.items():
                self._postings[term][chunk_id] = frequency

        average_length = (sum(lengths.values()) / max(1, len(lengths))) or 1
        self._length_norm = {
            chunk_id: _BM25_K1 * (1 - _BM25_B + _BM25_B * length / average_length)
            for chunk_id, length in lengths.items()
        }

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = Counter()
        for term in dict.fromkeys(tokenize(query)):
            matches = self._postings.get(term)
            if not matches:
                continue
            df = len(matches)
            idf = math.log1p((len(self._chunks) - df + 0.5) / (df + 0.5))
            for chunk_id, frequency in matches.items():
                scores[chunk_id] += idf * frequency * (_BM25_K1 + 1) / (frequency + self._length_norm[chunk_id])
        ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [self._chunks[chunk_id] for chunk_id in ranked_ids[:top_k]]


def rrf(vector_rankings: list[list[dict]], bm25_ranking: list[dict], top_k: int) -> list[dict]:
    """向量榜按查询顺序传入，每张榜分摊同一份向量权重。"""
    candidates = {}
    scores = Counter()
    vector_count = len(vector_rankings)
    for index, ranking in enumerate(vector_rankings):
        for rank, chunk in enumerate(ranking, 1):
            chunk_id = chunk["chunk_id"]
            if chunk_id not in candidates:
                candidates[chunk_id] = {**chunk, "channel_ranks": {"vector": [None] * vector_count, "bm25": None}}
            ranks = candidates[chunk_id]["channel_ranks"]["vector"]
            if ranks[index] is None:
                ranks[index] = rank
                scores[chunk_id] += 1 / (vector_count * (_RRF_K + rank))

    for rank, chunk in enumerate(bm25_ranking, 1):
        chunk_id = chunk["chunk_id"]
        if chunk_id not in candidates:
            candidates[chunk_id] = {**chunk, "score": None,
                                    "channel_ranks": {"vector": [None] * vector_count, "bm25": None}}
        ranks = candidates[chunk_id]["channel_ranks"]
        if ranks["bm25"] is None:
            ranks["bm25"] = rank
            scores[chunk_id] += 1 / (_RRF_K + rank)

    # 同分保留首次召回的顺序，原始问题的向量榜排在改写榜前面。
    ranked_ids = sorted(candidates, key=lambda chunk_id: -scores[chunk_id])
    return [{**candidates[chunk_id], "rrf_score": scores[chunk_id]} for chunk_id in ranked_ids[:top_k]]
