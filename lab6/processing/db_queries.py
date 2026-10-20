import numpy as np
import heapq
from scipy.sparse import csr_matrix
from .text_operations import vectorize_text
import faiss

def vectorize_query(query: str, vocab) -> csr_matrix:
    """Mapuje zapytanie na rzadki wektor wierszowy (1 x m)."""
    q_counts = vectorize_text(query)
    q_data = []
    q_indices = []

    for word, count in q_counts.items():
        if word in vocab:
            q_data.append(count)
            q_indices.append(vocab[word])

    m = len(vocab)
    q_vec = csr_matrix((q_data, ([0] * len(q_data), q_indices)), shape=(1, m))

    return q_vec


def format_results(scores, titles, urls, texts, result_len=10, is_pre_sorted=False):
    # Wyciąganie top-k przez kopiec i formatowanie do listy słowników
    # Łatwe do refactoringu :D
    if not is_pre_sorted:
        top_indices = heapq.nlargest(result_len, range(len(scores)), key=lambda i: scores[i])
    else:
        top_indices = scores

    return [
        {
            "title": titles[i],
            "url": urls[i],
            "snippet": " ".join((texts[i] or "").split())[:160]
        }
        for i in top_indices
    ]

def query_db_full(query: str, A, vocab, titles, urls, texts, result_len=10):
    """Wyszukiwanie w pełnej macierzy bez aproksymacji."""
    if not query:
        return []

    q_vec = vectorize_query(query, vocab)

    # Iloczyn skalarny
    dot_products = (q_vec * A).toarray()[0]

    # Wektorowe przekształcenie całej macierzy
    norm_q = np.sqrt(q_vec.power(2).sum())
    norm_docs = np.sqrt(np.asarray(A.power(2).sum(axis=0)).flatten())

    denominator = norm_q * norm_docs
    denominator[denominator == 0] = 1e-9

    scores = dot_products / denominator
    scores = np.clip(scores, -1.0, 1.0)

    return format_results(scores, titles, urls, texts, result_len)


def query_db_svd(query: str, svd_components, vocab, titles, urls, texts, result_len=10):
    """Wyszukiwanie z aproksymacją przez SVD w niższej wymiarowości."""
    if not query:
        return []

    q_vec = vectorize_query(query, vocab)

    U, S, hnsw_index = svd_components
    k = U.shape[1]
    if k == 0:
        return []

    # Projekcja zapytania do przestrzeni latentnej
    q_lat = np.asarray(q_vec.toarray().ravel() @ U[:, :k]).ravel()
    singular_values = S[:k].copy()
    singular_values[singular_values == 0] = 1e-9
    q_lat = q_lat / singular_values

    # formatowanie pod FAISS
    q_lat_2d = np.ascontiguousarray(q_lat.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(q_lat_2d)

    # zapytanie do grafu
    hnsw_index.hnsw.efSearch = 50
    distances, indices = hnsw_index.search(q_lat_2d, result_len)

    top_indices = indices[0]
    top_indices = top_indices[top_indices != -1]

    return format_results(top_indices, titles, urls, texts, result_len, is_pre_sorted=True)
